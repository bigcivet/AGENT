"""NEU-DET 多模型分类（70/15/15 划分、EfficientNet-B0 离线回退与模型切换）及 Streamlit 三页工作台 AppTest 验证。"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from generate_mock_neu_det import generate_mock_neu_det_dataset
from vision_agent.agent.tools import VisionToolkit
from vision_agent.config import DEFECT_CLASSES, VisionConfig
from vision_agent.decision.alert_engine import AlertEngine
from vision_agent.decision.rag import ProcessDocRAG
from vision_agent.memory.store import QualityMemoryStore
from vision_agent.vision.classifier import (
    EfficientNetB0DefectClassifier,
    ResNet18DefectClassifier,
    list_available_vision_models,
    load_vision_classifier,
)
from vision_agent.vision.dataset import NEUDetCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESS_DOCS_DIR = PROJECT_ROOT / "data" / "process_docs"
STREAMLIT_APP = PROJECT_ROOT / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def mock_neu_three_way_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("mock_neu_three_way")
    # 每类 20 张 -> 共 120 张，便于严格验证 70% / 15% / 15% 分层划分（84 / 18 / 18）
    generate_mock_neu_det_dataset(output_dir=out_dir, per_class=20, val_ratio=0.2, seed=99)
    return out_dir


def test_neu_three_way_split_70_15_15(mock_neu_three_way_dir: Path) -> None:
    catalog = NEUDetCatalog(mock_neu_three_way_dir)
    train_s, val_s, test_s = catalog.split_train_val_test(
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    assert len(train_s) == 84
    assert len(val_s) == 18
    assert len(test_s) == 18

    train_paths = {s.image_path for s in train_s}
    val_paths = {s.image_path for s in val_s}
    test_paths = {s.image_path for s in test_s}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    # 验证 6 类缺陷在训练/验证/测试集均完整覆盖
    assert {s.label for s in train_s} == set(DEFECT_CLASSES)
    assert {s.label for s in val_s} == set(DEFECT_CLASSES)
    assert {s.label for s in test_s} == set(DEFECT_CLASSES)


def test_efficientnet_b0_and_model_switching(
    mock_neu_three_way_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = NEUDetCatalog(mock_neu_three_way_dir)
    train_s, val_s, test_s = catalog.split_train_val_test(0.70, 0.15, 0.15, seed=42)

    # 模拟离线环境（下载预训练权重抛异常）时，EfficientNet-B0 应优雅回退为 weights=None
    import torchvision.models as tv_models

    original_eff = tv_models.efficientnet_b0

    def _offline_efficientnet_b0(*args, **kwargs):
        if kwargs.get("weights") is not None:
            raise RuntimeError("模拟离线环境无法下载预训练权重")
        return original_eff(*args, **kwargs)

    monkeypatch.setattr(tv_models, "efficientnet_b0", _offline_efficientnet_b0)

    eff = EfficientNetB0DefectClassifier(image_size=96, device="cpu")
    eff_metrics = eff.fit(
        catalog=catalog,
        epochs=1,
        batch_size=16,
        pretrained=True,  # 触发离线回退分支
        seed=42,
        custom_splits=(train_s, val_s, test_s),
    )
    assert eff_metrics["pretrained_weights_used"] is False
    assert eff_metrics["model_name"] == "efficientnet_b0"
    assert eff_metrics["split_mode"] == "three_way"
    assert eff_metrics["eval_split"] == "test"
    assert eff_metrics["test_samples"] == 18
    assert eff_metrics["avg_infer_time_ms"] > 0

    art_dir = tmp_path / "vision_multi"
    eff.save(art_dir)

    # 同时保存一个 ResNet18 以验证双模型并存与动态切换
    r18 = ResNet18DefectClassifier(image_size=96, device="cpu")
    r18.fit(
        catalog=catalog,
        epochs=1,
        batch_size=16,
        pretrained=False,
        seed=42,
        custom_splits=(train_s, val_s, test_s),
    )
    r18.save(art_dir)

    assert set(list_available_vision_models(art_dir)) == {"resnet18", "efficientnet_b0"}

    # 在 VisionToolkit 中动态切换 ResNet18 / EfficientNet-B0 并执行入库推理
    store = QualityMemoryStore(tmp_path / "switch_test.db")
    rag = ProcessDocRAG(PROCESS_DOCS_DIR)
    engine = AlertEngine(store=store, rag=rag, consecutive_threshold=3)
    toolkit = VisionToolkit(
        store=store,
        rag=rag,
        alert_engine=engine,
        classifier=load_vision_classifier("resnet18", art_dir, device="cpu"),
        config=VisionConfig(db_path=tmp_path / "switch_test.db", process_docs_dir=PROCESS_DOCS_DIR),
    )

    sample_img = test_s[0].image_path
    res_r18 = toolkit.inspect_image_and_record(sample_img, machine_id="M01", batch_id="B-R18")
    assert res_r18["prediction"]["model_name"] == "resnet18"

    toolkit.set_classifier(load_vision_classifier("efficientnet_b0", art_dir, device="cpu"))
    res_eff = toolkit.inspect_image_and_record(sample_img, machine_id="M01", batch_id="B-EFF")
    assert res_eff["prediction"]["model_name"] == "efficientnet_b0"
    assert res_eff["prediction"]["defect_type"] in DEFECT_CLASSES


def test_streamlit_three_pages_render_without_exception() -> None:
    """使用 Streamlit AppTest 端到端验证统一工作台 3 个页面均无报错渲染。"""
    at = AppTest.from_file(str(STREAMLIT_APP), default_timeout=30)
    at.run()
    assert not at.exception

    # 验证侧边栏 3 个页面选项完整存在
    options = at.sidebar.radio(key="workbench_page_selector").options
    assert len(options) == 3

    # 第 1 页：视觉质检智能体（默认选中）
    assert any("视觉质检智能体" in t.value for t in at.title)

    # 第 2 页：设备运维诊断智能体
    at.sidebar.radio(key="workbench_page_selector").set_value(options[1]).run()
    assert not at.exception
    assert any("设备运维诊断智能体" in t.value for t in at.title)

    # 第 3 页：数据分析与模型对比
    at.sidebar.radio(key="workbench_page_selector").set_value(options[2]).run()
    assert not at.exception
    assert any("数据分析与多模型对比" in t.value for t in at.title)
