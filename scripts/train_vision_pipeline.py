"""视觉质检智能体一键训练、离线评估与端到端联调脚本。

运行示例：
    python scripts/train_vision_pipeline.py --generate-mock --simulate-stream --epochs 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from generate_mock_neu_det import generate_mock_neu_det_dataset
from simulate_inspection_stream import simulate_inspection_stream
from vision_agent.agent.orchestrator import VisionQualityAgent
from vision_agent.agent.tools import VisionToolkit
from vision_agent.config import load_vision_config
from vision_agent.decision.alert_engine import AlertEngine
from vision_agent.decision.rag import ProcessDocRAG
from vision_agent.memory.store import QualityMemoryStore
from vision_agent.vision.classifier import ResNet18DefectClassifier
from vision_agent.vision.dataset import NEUDetCatalog


def main() -> None:
    cfg = load_vision_config()
    parser = argparse.ArgumentParser(description="NEU-DET ResNet18 缺陷分类训练与视觉质检智能体全流程验证")
    parser.add_argument("--data-dir", default=str(cfg.neu_det_dir), help="NEU-DET 数据目录")
    parser.add_argument("--artifact-dir", default=str(cfg.vision_artifact_dir), help="视觉模型保存目录")
    parser.add_argument("--db-path", default=str(cfg.db_path), help="SQLite 数字孪生数据库路径")
    parser.add_argument("--docs-dir", default=str(cfg.process_docs_dir), help="工艺文档目录")
    parser.add_argument("--generate-mock", action="store_true", help="数据为空时自动生成模拟 NEU-DET 图像")
    parser.add_argument("--per-class", type=int, default=60, help="生成模拟数据时每类图片数")
    parser.add_argument("--epochs", type=int, default=5, help="ResNet18 训练轮数（CPU 可配置）")
    parser.add_argument("--batch-size", type=int, default=16, help="训练批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--image-size", type=int, default=cfg.image_size, help="输入图像尺寸（默认 224）")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用 ImageNet 预训练权重")
    parser.add_argument("--freeze-backbone", action="store_true", help="冻结 ResNet18 浅层特征以加速 CPU 训练")
    parser.add_argument("--simulate-stream", action="store_true", help="同步模拟多机台多天质检流水并触发预警")
    parser.add_argument("--alert-threshold", type=int, default=cfg.alert_consecutive_threshold, help="连续缺陷预警阈值")
    args = parser.parse_args()

    catalog = NEUDetCatalog(args.data_dir)
    if not catalog.is_available:
        if args.generate_mock:
            print(f"[INFO] 目录 {args.data_dir} 暂无图像，正在生成 NEU-DET 模拟缺陷数据集（每类 {args.per_class} 张）...")
            stats = generate_mock_neu_det_dataset(args.data_dir, per_class=args.per_class)
            print(f"[INFO] 模拟数据集生成完成：{stats}")
            catalog.refresh()
        else:
            raise SystemExit(
                f"未在 {args.data_dir} 找到 NEU-DET 图像，请放入真实数据或追加 --generate-mock 参数。"
            )

    print(f"[INFO] 数据集扫描完成：{catalog.summary()}")
    print(
        f"[INFO] 开始训练 ResNet18 表面缺陷分类模型（epochs={args.epochs}, batch_size={args.batch_size}, "
        f"image_size={args.image_size}, pretrained={not args.no_pretrained}, freeze_backbone={args.freeze_backbone}）..."
    )

    classifier = ResNet18DefectClassifier(image_size=args.image_size)
    metrics = classifier.fit(
        catalog=catalog,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    ckpt = classifier.save(args.artifact_dir)

    print("\n================ ResNet18 视觉模型评估指标 ================")
    print(f"权重文件：{ckpt}")
    print(f"运行设备：{metrics['device']} | 使用预训练权重：{metrics['pretrained_weights_used']}")
    print(f"验证集准确率 (Accuracy)：{metrics['accuracy'] * 100:.2f}% | Macro-F1：{metrics['macro_f1'] * 100:.2f}%")
    print("各类指标：")
    for cls_name, item in metrics["per_class_metrics"].items():
        print(
            f"  - {cls_name:16s} ({item['cn_name']}): "
            f"P={item['precision']:.3f}, R={item['recall']:.3f}, F1={item['f1_score']:.3f} (n={item['support']})"
        )
    print("6x6 混淆矩阵：")
    print(json.dumps(metrics["confusion_matrix"]))

    if args.simulate_stream:
        sim_res = simulate_inspection_stream(
            db_path=args.db_path,
            process_docs_dir=args.docs_dir,
            neu_det_dir=args.data_dir,
            alert_threshold=args.alert_threshold,
            reset_db=True,
        )
        print(f"\n[INFO] 模拟产线质检流水完成：{sim_res}")

    # 端到端单图推理 + 入库 + 对话验证
    store = QualityMemoryStore(args.db_path)
    rag = ProcessDocRAG(args.docs_dir)
    engine = AlertEngine(store=store, rag=rag, consecutive_threshold=args.alert_threshold)
    toolkit = VisionToolkit(
        store=store, rag=rag, alert_engine=engine, classifier=classifier, config=cfg
    )

    sample = catalog.samples[0]
    inspect_res = toolkit.inspect_image_and_record(
        image_source=sample.image_path,
        machine_id="M01",
        batch_id="B-E2E-TEST",
        image_path=str(sample.image_path),
        notes="端到端脚本单图验证",
    )
    print(
        f"\n[INFO] 单图推理并自动入库成功：真实={sample.label}, "
        f"预测={inspect_res['prediction']['defect_type']} ({inspect_res['prediction']['defect_type_cn']}, "
        f"conf={inspect_res['prediction']['confidence']:.4f}), record_id={inspect_res['record_id']}"
    )

    agent = VisionQualityAgent(toolkit=toolkit, config=cfg)
    for q in ("最近哪种缺陷最多？", "3 号机台有没有异常趋势？"):
        resp = agent.chat(q, prefer_llm=False)
        print(f"\n>>> 测试问答：{q}")
        print(resp.reply[:500])


if __name__ == "__main__":
    main()
