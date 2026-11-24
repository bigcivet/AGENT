"""NEU-DET 金属表面瑕疵多模型分类对比一键脚本（ResNet18 vs EfficientNet-B0）。

对齐语雀《NEU 金属表面瑕疵分类参考文档》，自动化完成：
1. 按 70% / 15% / 15% 对全部样本做严格分层三划分（Train / Val / Test）；
2. 在同一数据划分与随机种子下训练评估 ResNet18 与 EfficientNet-B0（若无法下载预训练权重则优雅回退 `weights=None`）；
3. 输出测试集 Accuracy、Macro-F1、每类 Precision/Recall/F1、训练耗时、单图推理耗时与参数量；
4. 生成 JSON / Markdown 对比报告与高清 PNG 对比图及 6×6 混淆矩阵到 `reports/` 与 `reports/figures/`；
5. 默认持久化两套模型权重到 `artifacts/vision_models/`，供视觉质检智能体与工作台随时切换。

运行示例：
    python scripts/run_neu_benchmark.py --generate-mock --epochs 5 --batch-size 16 --freeze-backbone
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vision_agent.config import DEFECT_CLASSES, DEFECT_CN_MAP, load_vision_config
from vision_agent.vision.classifier import (
    VISION_MODEL_LABELS,
    EfficientNetB0DefectClassifier,
    ResNet18DefectClassifier,
)
from vision_agent.vision.dataset import NEUDetCatalog

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _plot_neu_comparison(models_metrics: dict[str, dict[str, Any]], out_path: Path) -> Path:
    """绘制 ResNet18 vs EfficientNet-B0 总体指标、推理耗时与每类 F1 对比图。"""
    keys = ["resnet18", "efficientnet_b0"]
    labels = [VISION_MODEL_LABELS[k] for k in keys]
    colors = {"resnet18": "#2b6cb0", "efficientnet_b0": "#38a169"}

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), dpi=180)

    # 子图 1：测试集 Accuracy & Macro-F1
    x = np.arange(len(keys))
    w = 0.30
    accs = [models_metrics[k]["accuracy"] * 100.0 for k in keys]
    f1s = [models_metrics[k]["macro_f1"] * 100.0 for k in keys]
    b1 = axes[0].bar(x - w / 2, accs, w, label="Test Accuracy (%)", color="#2b6cb0")
    b2 = axes[0].bar(x + w / 2, f1s, w, label="Test Macro-F1 (%)", color="#ed8936")
    axes[0].set_title("总体分类性能对比（70/15/15 独立测试集）")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylim(0, 114)
    axes[0].set_ylabel("指标 (%)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.25)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5)

    # 子图 2：每类缺陷 F1 对比
    cx = np.arange(len(DEFECT_CLASSES))
    cw = 0.36
    cls_cn = [DEFECT_CN_MAP.get(c, c) for c in DEFECT_CLASSES]
    r18_f1 = [models_metrics["resnet18"]["per_class_metrics"][c]["f1_score"] * 100.0 for c in DEFECT_CLASSES]
    eff_f1 = [models_metrics["efficientnet_b0"]["per_class_metrics"][c]["f1_score"] * 100.0 for c in DEFECT_CLASSES]
    axes[1].bar(cx - cw / 2, r18_f1, cw, label="ResNet18", color=colors["resnet18"])
    axes[1].bar(cx + cw / 2, eff_f1, cw, label="EfficientNet-B0", color=colors["efficientnet_b0"])
    axes[1].set_title("6 类金属表面缺陷 F1-Score 明细对比")
    axes[1].set_xticks(cx)
    axes[1].set_xticklabels(cls_cn, rotation=22, ha="right", fontsize=8.5)
    axes[1].set_ylim(0, 112)
    axes[1].set_ylabel("F1-Score (%)")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.25)

    # 子图 3：模型规模（参数量 M）与单图推理耗时（ms）
    params_m = [models_metrics[k]["total_params"] / 1e6 for k in keys]
    infer_ms = [models_metrics[k]["avg_infer_time_ms"] for k in keys]
    ax_l = axes[2]
    ax_r = ax_l.twinx()
    b3 = ax_l.bar(x - w / 2, params_m, w, label="参数量 (M)", color="#4a5568")
    b4 = ax_r.bar(x + w / 2, infer_ms, w, label="单图推理耗时 (ms)", color="#e53e3e")
    ax_l.set_title("模型轻量化与 CPU 推理效率对比")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, fontsize=10)
    ax_l.set_ylim(0, max(params_m) * 1.26)
    ax_r.set_ylim(0, max(infer_ms) * 1.26)
    ax_l.set_ylabel("参数量 (Millions)", color="#4a5568")
    ax_r.set_ylabel("单图平均推理耗时 (ms)", color="#e53e3e")
    ax_l.grid(axis="y", alpha=0.25)
    for bar in b3:
        h = bar.get_height()
        ax_l.text(bar.get_x() + bar.get_width() / 2, h + max(params_m) * 0.025, f"{h:.2f}M", ha="center", va="bottom", fontsize=8.5)
    for bar in b4:
        h = bar.get_height()
        ax_r.text(bar.get_x() + bar.get_width() / 2, h + max(infer_ms) * 0.025, f"{h:.1f}ms", ha="center", va="bottom", fontsize=8.5, color="#c53030")
    l1, lb1 = ax_l.get_legend_handles_labels()
    l2, lb2 = ax_r.get_legend_handles_labels()
    ax_l.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper right")

    fig.suptitle("NEU-DET 金属表面缺陷多模型分类基准对比（ResNet18 vs EfficientNet-B0）", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_neu_confusion_matrices(models_metrics: dict[str, dict[str, Any]], out_path: Path) -> Path:
    """绘制 ResNet18 与 EfficientNet-B0 的 6×6 测试集混淆矩阵并排热力图。"""
    keys = ["resnet18", "efficientnet_b0"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=180)
    cls_labels = [DEFECT_CN_MAP.get(c, c) for c in DEFECT_CLASSES]

    for ax, key in zip(axes, keys):
        m = models_metrics[key]
        cm = np.asarray(m["confusion_matrix"], dtype=np.int64)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(
            f"{VISION_MODEL_LABELS[key]}\nTest Acc={m['accuracy']*100:.2f}% | Macro-F1={m['macro_f1']*100:.2f}%",
            fontsize=10.5,
        )
        ax.set_xticks(range(len(cls_labels)))
        ax.set_yticks(range(len(cls_labels)))
        ax.set_xticklabels(cls_labels, rotation=28, ha="right", fontsize=8.5)
        ax.set_yticklabels(cls_labels, fontsize=8.5)
        ax.set_xlabel("预测缺陷类别")
        ax.set_ylabel("真实缺陷类别")
        thresh = cm.max() * 0.5 if cm.size > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9,
                )
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("NEU-DET 独立测试集 6×6 混淆矩阵对比（ResNet18 vs EfficientNet-B0）", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _render_neu_markdown(report: dict[str, Any]) -> str:
    models_m = report["models"]
    split = report["split_summary"]
    lines = [
        "# NEU-DET 金属表面瑕疵多模型分类对比报告",
        "",
        f"- **生成时间**：{report['generated_at']}",
        f"- **数据目录**：`{report['data_dir']}`",
        f"- **数据集划分（70% / 15% / 15% 分层）**：总计 `{split['total_images']}` 张（训练集 `{split['train_images']}` / 验证集 `{split['val_images']}` / 测试集 `{split['test_images']}`）",
        f"- **训练配置**：epochs={report['config']['epochs']}，batch_size={report['config']['batch_size']}，image_size={report['config']['image_size']}，freeze_backbone={report['config']['freeze_backbone']}",
        "",
        "## 一、总体性能与效率对比（独立测试集）",
        "",
        "| 模型 | 预训练权重 | 参数量 (M) | Test Accuracy | Test Macro-F1 | 训练耗时 (s) | 单图推理耗时 (ms) |",
        "| --- | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("resnet18", "efficientnet_b0"):
        m = models_m[key]
        lines.append(
            f"| {VISION_MODEL_LABELS[key]} | {'是' if m['pretrained_weights_used'] else '否(离线回退)'} | "
            f"{m['total_params']/1e6:.2f} | {m['accuracy']*100:.2f}% | {m['macro_f1']*100:.2f}% | "
            f"{m['train_time_seconds']:.2f} | {m['avg_infer_time_ms']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 二、6 类金属表面缺陷逐类 Precision / Recall / F1 明细",
            "",
            "| 缺陷类别 | ResNet18 P | ResNet18 R | ResNet18 F1 | EfficientNet-B0 P | EfficientNet-B0 R | EfficientNet-B0 F1 | 测试样本数 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cls in DEFECT_CLASSES:
        r = models_m["resnet18"]["per_class_metrics"][cls]
        e = models_m["efficientnet_b0"]["per_class_metrics"][cls]
        lines.append(
            f"| {DEFECT_CN_MAP.get(cls, cls)} (`{cls}`) | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1_score']:.4f} | "
            f"{e['precision']:.4f} | {e['recall']:.4f} | {e['f1_score']:.4f} | {r['support']} |"
        )

    lines.extend(
        [
            "",
            "## 三、生成的对比图表",
            "",
            "- `reports/figures/neu_model_comparison.png`：ResNet18 vs EfficientNet-B0 准确率、Macro-F1、每类 F1 及推理耗时对比",
            "- `reports/figures/neu_confusion_matrices.png`：ResNet18 vs EfficientNet-B0 测试集 6×6 混淆矩阵对比",
        ]
    )
    return "\n".join(lines) + "\n"


def run_neu_benchmark(
    data_dir: str | Path,
    report_dir: str | Path = PROJECT_ROOT / "reports",
    artifact_dir: str | Path = PROJECT_ROOT / "artifacts" / "vision_models",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    image_size: int = 224,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    seed: int = 42,
    save_models: bool = True,
) -> dict[str, Any]:
    """在 70/15/15 划分下对比 ResNet18 与 EfficientNet-B0，生成报告与 PNG 图表。"""
    catalog = NEUDetCatalog(data_dir)
    if not catalog.is_available:
        raise RuntimeError(f"未在 `{data_dir}` 扫描到有效的 NEU-DET 缺陷图像。")

    train_s, val_s, test_s = catalog.split_train_val_test(
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
    )

    report_root = Path(report_dir)
    figures_dir = report_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    models_metrics: dict[str, dict[str, Any]] = {}
    for model_key, cls_type in (
        ("resnet18", ResNet18DefectClassifier),
        ("efficientnet_b0", EfficientNetB0DefectClassifier),
    ):
        print(f"[INFO] 正在训练并评估 {VISION_MODEL_LABELS[model_key]} (epochs={epochs}, image_size={image_size})...")
        clf = cls_type(image_size=image_size)
        metrics = clf.fit(
            catalog=catalog,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            seed=seed,
            custom_splits=(train_s, val_s, test_s),
        )
        if save_models:
            clf.save(artifact_dir)
        models_metrics[model_key] = metrics

    fig_cmp = _plot_neu_comparison(models_metrics, figures_dir / "neu_model_comparison.png")
    fig_cm = _plot_neu_confusion_matrices(models_metrics, figures_dir / "neu_confusion_matrices.png")

    report: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(Path(data_dir).resolve()),
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "image_size": image_size,
            "pretrained_requested": pretrained,
            "freeze_backbone": freeze_backbone,
            "seed": seed,
        },
        "split_summary": {
            "ratios": [0.70, 0.15, 0.15],
            "total_images": len(train_s) + len(val_s) + len(test_s),
            "train_images": len(train_s),
            "val_images": len(val_s),
            "test_images": len(test_s),
        },
        "models": models_metrics,
        "figures": {
            "model_comparison": fig_cmp.name,
            "confusion_matrices": fig_cm.name,
        },
    }

    (report_root / "neu_benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_root / "neu_benchmark_report.md").write_text(
        _render_neu_markdown(report), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    cfg = load_vision_config()
    parser = argparse.ArgumentParser(description="NEU-DET 金属表面缺陷多模型对比脚本（ResNet18 vs EfficientNet-B0）")
    parser.add_argument("--data-dir", default=str(cfg.neu_det_dir), help="NEU-DET 数据目录")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "reports"), help="报告与图表输出目录")
    parser.add_argument("--artifact-dir", default=str(cfg.vision_artifact_dir), help="视觉模型保存目录")
    parser.add_argument("--generate-mock", action="store_true", help="数据目录为空时自动生成模拟 NEU-DET 图像")
    parser.add_argument("--per-class", type=int, default=60, help="生成模拟数据时每类图片数")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数（CPU 可配置）")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--image-size", type=int, default=cfg.image_size, help="输入图像尺寸（默认 224）")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用 ImageNet 预训练权重")
    parser.add_argument("--freeze-backbone", action="store_true", help="冻结主干浅层特征以加速 CPU 训练")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no-save-models", action="store_true", help="仅输出报告，不保存模型权重")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = NEUDetCatalog(args.data_dir)
    if not catalog.is_available:
        if args.generate_mock:
            from generate_mock_neu_det import generate_mock_neu_det_dataset

            print(f"[INFO] 目录 {args.data_dir} 为空，正在生成模拟 NEU-DET 数据集（每类 {args.per_class} 张）...")
            generate_mock_neu_det_dataset(output_dir=args.data_dir, per_class=args.per_class, seed=args.seed)
        else:
            raise SystemExit(
                f"[错误] 未在 `{args.data_dir}` 找到 NEU-DET 图像，请放入真实数据或追加 `--generate-mock`。"
            )

    report = run_neu_benchmark(
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        artifact_dir=args.artifact_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        seed=args.seed,
        save_models=not args.no_save_models,
    )

    print("\n================ NEU-DET 多模型测试集对比结果 ================")
    sp = report["split_summary"]
    print(f"数据划分 (70/15/15)：Train={sp['train_images']} | Val={sp['val_images']} | Test={sp['test_images']}")
    for k in ("resnet18", "efficientnet_b0"):
        m = report["models"][k]
        print(
            f"  - {VISION_MODEL_LABELS[k]:16s}: "
            f"Test Acc={m['accuracy']*100:6.2f}% | Macro-F1={m['macro_f1']*100:6.2f}% | "
            f"Train={m['train_time_seconds']:.2f}s | Infer={m['avg_infer_time_ms']:.2f}ms/img | "
            f"Pretrained={m['pretrained_weights_used']}"
        )
    print(f"\n[OK] 报告与图表已输出至：{Path(args.report_dir).resolve()}")


if __name__ == "__main__":
    main()
