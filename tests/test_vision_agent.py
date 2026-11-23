"""视觉质检智能体单元测试与端到端集成测试。

覆盖：
1. NEU-DET 模拟缺陷图像生成、VOC XML 解析、数据集扫描与 YOLO 格式导出；
2. ResNet18 缺陷分类器训练、评估指标（准确率、混淆矩阵）、模型持久化与单图推理；
3. SQLite 数字孪生记忆库写入、统计、趋势分析与受限只读 SQL 安全校验；
4. 本地工艺文档 RAG 检索（`data/process_docs/`）与连续缺陷主动预警引擎；
5. 视觉质检 Agent 自然语言查询（“最近哪种缺陷最多？”、“3 号机台有没有异常趋势？”等）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from equipment_agent.config import AppConfig
from generate_mock_neu_det import generate_mock_neu_det_dataset
from simulate_inspection_stream import simulate_inspection_stream
from vision_agent.agent.orchestrator import VisionQualityAgent
from vision_agent.agent.tools import VisionToolkit
from vision_agent.config import DEFECT_CLASSES, VisionConfig
from vision_agent.decision.alert_engine import AlertEngine
from vision_agent.decision.rag import ProcessDocRAG
from vision_agent.memory.store import InspectionRecord, QualityMemoryStore
from vision_agent.vision.classifier import ResNet18DefectClassifier
from vision_agent.vision.dataset import NEUDetCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESS_DOCS_DIR = PROJECT_ROOT / "data" / "process_docs"


@pytest.fixture(scope="module")
def mock_neu_det_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("mock_neu_det")
    generate_mock_neu_det_dataset(output_dir=out_dir, per_class=10, val_ratio=0.2, seed=7)
    return out_dir


def test_neu_det_loader_and_yolo_export(mock_neu_det_dir: Path, tmp_path: Path) -> None:
    catalog = NEUDetCatalog(mock_neu_det_dir)
    assert catalog.is_available
    summary = catalog.summary()
    assert summary["total_images"] == 60
    assert summary["annotated_images"] == 60
    assert set(summary["class_counts"].keys()) == set(DEFECT_CLASSES)

    train_s, val_s = catalog.split_samples()
    assert len(train_s) == 48
    assert len(val_s) == 12
    assert all(len(s.bboxes) >= 1 for s in train_s)

    yaml_path = catalog.export_yolo_dataset(tmp_path / "yolo_export")
    assert yaml_path.exists()
    content = yaml_path.read_text(encoding="utf-8")
    assert "nc: 6" in content
    assert "rolled-in_scale" in content


def test_resnet18_train_save_load_predict(mock_neu_det_dir: Path, tmp_path: Path) -> None:
    catalog = NEUDetCatalog(mock_neu_det_dir)
    # 测试环境使用无预训练权重 + 1 epoch + 较小输入尺寸，保证秒级快速验证全流程
    clf = ResNet18DefectClassifier(image_size=96, device="cpu")
    metrics = clf.fit(
        catalog=catalog,
        epochs=1,
        batch_size=12,
        pretrained=False,
        seed=42,
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 6
    assert len(metrics["confusion_matrix"][0]) == 6

    art_dir = tmp_path / "vision_models"
    ckpt = clf.save(art_dir)
    assert ckpt.exists()
    assert (art_dir / ResNet18DefectClassifier.METRICS_FILENAME).exists()

    loaded = ResNet18DefectClassifier.load(art_dir, device="cpu")
    pred = loaded.predict_image(catalog.samples[0].image_path)
    assert pred.defect_type in DEFECT_CLASSES
    assert 0.0 <= pred.confidence <= 1.0
    assert abs(sum(pred.probabilities.values()) - 1.0) < 0.02


def test_memory_store_readonly_sql_and_alerts(tmp_path: Path) -> None:
    db_path = tmp_path / "vision_test.db"
    store = QualityMemoryStore(db_path)
    rag = ProcessDocRAG(PROCESS_DOCS_DIR)
    assert rag.is_ready
    assert len(rag.list_documents()) >= 3

    engine = AlertEngine(store=store, rag=rag, consecutive_threshold=3)

    # 同一机台连续写入 3 次划痕，前 2 次不触发，第 3 次应触发预警并关联工艺文档
    for i in range(3):
        store.record_inspection(
            InspectionRecord(
                machine_id="M03",
                batch_id=f"B-TEST-{i}",
                defect_type="scratches",
                confidence=0.92,
                inspected_at=f"2026-11-23 10:0{i}:00",
            )
        )
        alert = engine.evaluate_machine("M03")
        if i < 2:
            assert alert is None
        else:
            assert alert is not None
            assert alert.machine_id == "M03"
            assert alert.defect_type == "scratches"
            assert alert.consecutive_count == 3
            assert len(alert.doc_references) >= 1

    # 只读 SQL 正常查询通过
    ok = store.execute_readonly_sql("SELECT machine_id, COUNT(*) AS cnt FROM inspections GROUP BY machine_id")
    assert ok["row_count"] == 1
    assert ok["rows"][0]["cnt"] == 3

    # 危险写入/删除 SQL 必须被拦截
    for dangerous in (
        "DELETE FROM inspections",
        "DROP TABLE alerts",
        "SELECT * FROM inspections; DROP TABLE alerts",
        "UPDATE inspections SET defect_type='patches'",
    ):
        with pytest.raises(ValueError):
            store.execute_readonly_sql(dangerous)


def test_vision_agent_nl_queries_and_rag(tmp_path: Path) -> None:
    db_path = tmp_path / "sim_vision.db"
    sim_res = simulate_inspection_stream(
        db_path=db_path,
        process_docs_dir=PROCESS_DOCS_DIR,
        days=7,
        alert_threshold=3,
        reset_db=True,
        seed=42,
    )
    assert sim_res["inserted_records"] > 100
    assert sim_res["alerts_triggered"] >= 2

    store = QualityMemoryStore(db_path)
    rag = ProcessDocRAG(PROCESS_DOCS_DIR)
    engine = AlertEngine(store=store, rag=rag, consecutive_threshold=3)
    offline_cfg = VisionConfig(
        db_path=db_path,
        process_docs_dir=PROCESS_DOCS_DIR,
        app_config=AppConfig(openai_api_key=""),
    )
    toolkit = VisionToolkit(store=store, rag=rag, alert_engine=engine, config=offline_cfg)
    agent = VisionQualityAgent(toolkit=toolkit, config=offline_cfg)

    # 1. 最近哪种缺陷最多？
    r1 = agent.chat("最近哪种缺陷最多？")
    assert r1.mode == "rule_router"
    assert any(tc.tool_name == "get_defect_summary" for tc in r1.tool_calls)
    assert "缺陷" in r1.reply

    # 2. 3 号机台有没有异常趋势？
    r2 = agent.chat("3 号机台有没有异常趋势？")
    assert any(tc.tool_name == "analyze_machine_trend" for tc in r2.tool_calls)
    assert "M03" in r2.reply
    assert "异常" in r2.reply
    assert "《" in r2.reply  # 引用工艺文档

    # 3. 主动预警查询
    r3 = agent.chat("当前有哪些连续缺陷预警？")
    assert any(tc.tool_name == "list_quality_alerts" for tc in r3.tool_calls)
    assert "M03" in r3.reply

    # 4. 工艺知识库 RAG 问答
    r4 = agent.chat("压入氧化皮的工艺原因和处置措施是什么？")
    assert any(tc.tool_name == "search_process_docs" for tc in r4.tool_calls)
    assert "除鳞" in r4.reply
    assert "02_金属表面缺陷成因与处置手册.md" in r4.reply
