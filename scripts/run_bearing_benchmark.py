"""西交大 XJTU-SY 滚动轴承数据分析与多模型基准对比一键脚本。

对齐语雀实训指南与特征提取 / RUL 归一化参考文档，自动化完成以下实验与图表交付：
1. 特征工程分析：54 维双通道时频特征相关性分析 + 可选 PCA 降维（95% 累计方差）效果对比；
2. 空间划分对比（按轴承单元 7:3 划分，空间不重叠）：RF vs 1D-CNN vs LSTM 故障分类；
3. 时间窗划分对比（训练集 20%~55%，测试集 55%~70%）+ 评估指标随生命周期推移变化趋势；
4. RUL 剩余寿命预测归一化对比：Max-based 相对归一化 [0,1] vs 工况分组归一化 vs 绝对分钟；
5. 自动生成 JSON / Markdown 对比报告与高清 PNG 图表到 `reports/` 和 `reports/figures/`；
6. 可选同步持久化设备运维智能体可切换的 RF / 1D-CNN / LSTM 模型到 `artifacts/models/`。

运行示例：
    python scripts/run_bearing_benchmark.py --generate-mock --epochs 35 --batch-size 32
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
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from equipment_agent.config import load_config
from equipment_agent.data.loader import (
    XJTUSYDataLoader,
    evaluate_predictions_over_time_bins,
    split_dataframe_by_time_window,
)
from equipment_agent.features.analysis import (
    PCADimensionalityReducer,
    compute_feature_correlation,
)
from equipment_agent.features.extractor import FEATURE_COLUMNS, build_feature_dataset
from equipment_agent.models.classifier import RandomForestFaultClassifier
from equipment_agent.models.deep_classifiers import (
    CNN1DFaultClassifier,
    FAULT_CLASSIFIER_LABELS,
    LSTMFaultClassifier,
)
from equipment_agent.models.rul import compare_rul_normalization_strategies
from equipment_agent.models.trainer import TrainingPipeline

# 统一配置中文绘图字体
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _plot_model_comparison(
    spatial_results: dict[str, dict[str, Any]],
    time_results: dict[str, dict[str, Any]],
    out_path: Path,
) -> Path:
    """绘制 RF / 1D-CNN / LSTM 准确率、Macro-F1 与耗时柱状对比图。"""
    model_keys = ["rf", "cnn1d", "lstm"]
    names = [FAULT_CLASSIFIER_LABELS[k] for k in model_keys]
    x = np.arange(len(model_keys))
    width = 0.32

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=180)

    # 子图 1：空间 7:3 划分（按轴承单元）
    sp_acc = [spatial_results[k]["accuracy"] * 100.0 for k in model_keys]
    sp_f1 = [spatial_results[k]["macro_f1"] * 100.0 for k in model_keys]
    b1 = axes[0].bar(x - width / 2, sp_acc, width, label="Accuracy (%)", color="#2b6cb0")
    b2 = axes[0].bar(x + width / 2, sp_f1, width, label="Macro-F1 (%)", color="#ed8936")
    axes[0].set_title("按轴承单元 7:3 空间划分对比")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("指标 (%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    # 子图 2：时间窗划分（20%~55% 训练 -> 55%~70% 测试）
    tw_acc = [time_results[k]["accuracy"] * 100.0 for k in model_keys]
    tw_f1 = [time_results[k]["macro_f1"] * 100.0 for k in model_keys]
    b3 = axes[1].bar(x - width / 2, tw_acc, width, label="Accuracy (%)", color="#319795")
    b4 = axes[1].bar(x + width / 2, tw_f1, width, label="Macro-F1 (%)", color="#805ad5")
    axes[1].set_title("时间窗划分对比 (20%~55% → 55%~70%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylim(0, 112)
    axes[1].set_ylabel("指标 (%)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    # 子图 3：训练耗时 (s) 与单样本推理耗时 (ms)
    train_s = [spatial_results[k]["train_time_seconds"] for k in model_keys]
    infer_ms = [spatial_results[k]["infer_time_ms_per_sample"] for k in model_keys]
    ax_left = axes[2]
    ax_right = ax_left.twinx()
    b5 = ax_left.bar(x - width / 2, train_s, width, label="训练耗时 (s)", color="#4a5568")
    b6 = ax_right.bar(x + width / 2, infer_ms, width, label="单样本推理 (ms)", color="#e53e3e")
    ax_left.set_title("计算开销对比（CPU）")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(names, fontsize=9)
    ax_left.set_ylim(0, max(train_s) * 1.28)
    ax_right.set_ylim(0, max(infer_ms) * 1.28)
    ax_left.set_ylabel("训练耗时 (s)", color="#4a5568")
    ax_right.set_ylabel("单样本推理耗时 (ms)", color="#e53e3e")
    ax_left.grid(axis="y", alpha=0.25)
    for bar in b5:
        h = bar.get_height()
        ax_left.text(bar.get_x() + bar.get_width() / 2, h + max(train_s) * 0.025, f"{h:.2f}s", ha="center", va="bottom", fontsize=8)
    for bar in b6:
        h = bar.get_height()
        ax_right.text(bar.get_x() + bar.get_width() / 2, h + max(infer_ms) * 0.025, f"{h:.3f}ms", ha="center", va="bottom", fontsize=7.5, color="#c53030")
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.suptitle("XJTU-SY 滚动轴承故障分类多模型基准对比（RF vs 1D-CNN vs LSTM）", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_confusion_matrices(
    spatial_results: dict[str, dict[str, Any]],
    out_path: Path,
) -> Path:
    """绘制 RF / 1D-CNN / LSTM 混淆矩阵并排对比热力图。"""
    model_keys = ["rf", "cnn1d", "lstm"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), dpi=180)

    for ax, key in zip(axes, model_keys):
        res = spatial_results[key]
        cm = np.asarray(res["confusion_matrix"], dtype=np.int64)
        labels = res["labels"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(
            f"{FAULT_CLASSIFIER_LABELS[key]}\nAcc={res['accuracy']*100:.1f}% | Macro-F1={res['macro_f1']*100:.1f}%",
            fontsize=10,
        )
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("预测故障类型")
        ax.set_ylabel("真实故障类型")
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

    fig.suptitle("XJTU-SY 轴承故障分类混淆矩阵对比（按轴承单元 7:3 空间划分）", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_accuracy_over_time(
    time_curves: dict[str, list[dict[str, Any]]],
    out_path: Path,
) -> Path:
    """绘制三模型在轴承全生命周期各阶段准确率与 Macro-F1 随时间推移曲线。"""
    colors = {"rf": "#2b6cb0", "cnn1d": "#e53e3e", "lstm": "#38a169"}
    markers = {"rf": "o", "cnn1d": "s", "lstm": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), dpi=180)
    for key in ("rf", "cnn1d", "lstm"):
        bins = time_curves.get(key, [])
        if not bins:
            continue
        x_labels = [b["bin_label"] for b in bins]
        accs = [b["accuracy"] * 100.0 for b in bins]
        f1s = [b["macro_f1"] * 100.0 for b in bins]
        axes[0].plot(
            x_labels, accs, marker=markers[key], linewidth=2.2, label=FAULT_CLASSIFIER_LABELS[key], color=colors[key]
        )
        axes[1].plot(
            x_labels, f1s, marker=markers[key], linewidth=2.2, label=FAULT_CLASSIFIER_LABELS[key], color=colors[key]
        )

    for ax, metric_name in zip(axes, ("准确率 Accuracy (%)", "宏平均 F1 Macro-F1 (%)")):
        ax.set_title(f"时间窗模型（20%~55% 训练）随生命周期推移 {metric_name}")
        ax.set_xlabel("轴承全寿命周期区间（Life Ratio）")
        ax.set_ylabel(metric_name)
        ax.set_ylim(0, 108)
        ax.grid(alpha=0.28)
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("轴承故障识别性能随退化时间推移变化趋势分析", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_rul_degradation(
    test_df: pd.DataFrame,
    rul_comparison: dict[str, Any],
    out_path: Path,
) -> Path:
    """绘制 RUL 归一化策略误差对比图 + 典型测试轴承退化预测曲线。"""
    strategies = ["max_based", "group_minmax", "absolute"]
    labels = [rul_comparison[s]["label"] for s in strategies]
    rmses = [rul_comparison[s]["relative_scale"]["rmse"] for s in strategies]
    maes = [rul_comparison[s]["relative_scale"]["mae"] for s in strategies]
    r2s = [rul_comparison[s]["relative_scale"]["r2"] for s in strategies]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), dpi=180)

    # 左图：三种归一化策略同尺度误差与 R² 对比
    x = np.arange(len(strategies))
    w = 0.25
    b1 = axes[0].bar(x - w, rmses, w, label="RMSE (相对尺度)", color="#e53e3e")
    b2 = axes[0].bar(x, maes, w, label="MAE (相对尺度)", color="#dd6b20")
    b3 = axes[0].bar(x + w, r2s, w, label="R² 决定系数", color="#2b6cb0")
    axes[0].set_title("RUL 归一化策略测试集性能对比（统一 [0,1] 相对尺度）")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8.5)
    axes[0].set_ylim(min(0.0, min(r2s) - 0.1), max(1.12, max(rmses) + 0.15))
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    for bar in list(b1) + list(b2) + list(b3):
        h = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h + (0.02 if h >= 0 else -0.05),
            f"{h:.3f}",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=7.5,
        )

    # 右图：选取第一个测试轴承的全生命周期 RUL 退化预测曲线
    sample_bearing_id = str(test_df["bearing_id"].iloc[0])
    mask = (test_df["bearing_id"] == sample_bearing_id).to_numpy()
    sub = test_df.loc[mask].sort_values("file_index")
    idx_order = sub.index.to_numpy()
    pos_map = {orig_idx: pos for pos, orig_idx in enumerate(test_df.index)}
    ordered_pos = [pos_map[i] for i in idx_order]

    time_axis = sub["file_index"].to_numpy()
    true_rel = sub["rul_normalized"].to_numpy()
    pred_max = np.asarray(rul_comparison["max_based"]["predictions_relative"])[ordered_pos]
    pred_grp = np.asarray(rul_comparison["group_minmax"]["predictions_relative"])[ordered_pos]
    pred_abs = np.asarray(rul_comparison["absolute"]["predictions_relative"])[ordered_pos]

    axes[1].plot(time_axis, true_rel, label="真实相对 RUL", color="#1a202c", linewidth=2.4)
    axes[1].plot(time_axis, pred_max, label="Max-based 相对归一化预测", color="#2b6cb0", linestyle="--", marker="o", markersize=3.5)
    axes[1].plot(time_axis, pred_grp, label="工况分组归一化预测", color="#38a169", linestyle="-.", marker="s", markersize=3)
    axes[1].plot(time_axis, pred_abs, label="绝对分钟回归折算预测", color="#e53e3e", linestyle=":", marker="^", markersize=3)
    axes[1].set_title(f"典型测试轴承 {sample_bearing_id} 全生命周期 RUL 退化预测曲线")
    axes[1].set_xlabel("运行时间（分钟）")
    axes[1].set_ylabel("相对剩余寿命比例 [0, 1]")
    axes[1].set_ylim(-0.05, 1.08)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("XJTU-SY 轴承剩余寿命（RUL）归一化策略对比与退化曲线", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_feature_correlation_and_pca(
    corr_report,
    pca_summary: dict[str, Any],
    out_path: Path,
) -> Path:
    """绘制 Top-12 RUL 敏感特征相关性热力图与 PCA 累计方差解释率曲线。"""
    top_feats = [item["feature"] for item in corr_report.rul_correlations[:12]]
    if not top_feats:
        top_feats = corr_report.feature_columns[:12]
    sub_corr = corr_report.corr_matrix.loc[top_feats, top_feats]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), dpi=180)

    # 左图：Top-12 特征相关性热力图
    im = axes[0].imshow(sub_corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0].set_title("Top-12 退化敏感时频特征相关性热力图")
    axes[0].set_xticks(range(len(top_feats)))
    axes[0].set_yticks(range(len(top_feats)))
    axes[0].set_xticklabels(top_feats, rotation=40, ha="right", fontsize=7.5)
    axes[0].set_yticklabels(top_feats, fontsize=7.5)
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # 右图：PCA 累计方差解释率曲线
    cum_var = [v * 100.0 for v in pca_summary["cumulative_variance_ratio"]]
    indiv_var = [v * 100.0 for v in pca_summary["explained_variance_ratio"]]
    pcs = np.arange(1, len(cum_var) + 1)
    axes[1].bar(pcs, indiv_var, alpha=0.55, color="#4299e1", label="单主成分方差解释率 (%)")
    axes[1].plot(pcs, cum_var, color="#e53e3e", marker="o", linewidth=2.0, label="累计方差解释率 (%)")
    axes[1].axhline(95.0, color="#2f855a", linestyle="--", linewidth=1.3, label="95% 阈值线")
    axes[1].set_title(
        f"54 维时频特征 PCA 降维（{pca_summary['input_dim']} 维 → {pca_summary['output_dim']} 维，"
        f"解释 {pca_summary['total_explained_variance']*100:.1f}% 方差）"
    )
    axes[1].set_xlabel("主成分序号 (PC)")
    axes[1].set_ylabel("方差解释率 (%)")
    axes[1].set_ylim(0, 108)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8.5)

    fig.suptitle("XJTU-SY 轴承 54 维时频特征相关性分析与 PCA 降维", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _render_markdown_report(report: dict[str, Any]) -> str:
    """将轴承基准对比 JSON 报告渲染为结构化 Markdown 文档。"""
    sp = report["spatial_split_70_30"]["models"]
    tw = report["time_window_split"]["models"]
    rul = report["rul_comparison"]
    pca = report["feature_engineering"]["pca"]
    pca_rf = report["feature_engineering"]["rf_with_pca"]

    lines = [
        "# XJTU-SY 滚动轴承数据分析与多模型对比报告",
        "",
        f"- **生成时间**：{report['generated_at']}",
        f"- **数据目录**：`{report['data_dir']}`",
        f"- **轴承总数**：{report['dataset_summary']['total_bearings']} 个（有效样本 {report['dataset_summary']['total_samples']} 条，54 维时频特征）",
        f"- **深度学习配置**：epochs={report['config']['epochs']}，batch_size={report['config']['batch_size']}，seed={report['config']['seed']}",
        "",
        "## 一、特征相关性分析与 PCA 降维",
        "",
        f"- 原始特征维度：**{pca['input_dim']} 维**，发现高相关冗余特征对（|r| ≥ 0.90）：**{report['feature_engineering']['correlation']['high_corr_pair_count']} 对**。",
        f"- PCA 降维（95% 累计方差阈值）：由 **{pca['input_dim']} 维** 压缩至 **{pca['output_dim']} 维**（累计解释方差 `{pca['total_explained_variance']*100:.2f}%`）。",
        f"- 空间划分下随机森林对比：原始 54 维 Accuracy=`{sp['rf']['accuracy']*100:.2f}%` / Macro-F1=`{sp['rf']['macro_f1']*100:.2f}%` vs PCA 降维后 Accuracy=`{pca_rf['accuracy']*100:.2f}%` / Macro-F1=`{pca_rf['macro_f1']*100:.2f}%`。",
        "",
        "## 二、故障分类多模型对比（按轴承单元 7:3 空间不重叠划分）",
        "",
        f"- 训练轴承（{len(report['spatial_split_70_30']['train_bearings'])} 个）：`{', '.join(report['spatial_split_70_30']['train_bearings'])}`",
        f"- 测试轴承（{len(report['spatial_split_70_30']['test_bearings'])} 个）：`{', '.join(report['spatial_split_70_30']['test_bearings'])}`",
        "",
        "| 模型 | Accuracy | Macro-F1 | 训练耗时 (s) | 单样本推理耗时 (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ("rf", "cnn1d", "lstm"):
        m = sp[key]
        lines.append(
            f"| {FAULT_CLASSIFIER_LABELS[key]} | {m['accuracy']*100:.2f}% | {m['macro_f1']*100:.2f}% | "
            f"{m['train_time_seconds']:.3f} | {m['infer_time_ms_per_sample']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 三、时间窗划分对比（训练集 20%~55% → 测试集 55%~70%）",
            "",
            f"- 训练样本数：`{report['time_window_split']['train_samples']}` | 测试样本数：`{report['time_window_split']['test_samples']}`",
            "",
            "| 模型 | Accuracy | Macro-F1 | 训练耗时 (s) | 单样本推理耗时 (ms) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in ("rf", "cnn1d", "lstm"):
        m = tw[key]
        lines.append(
            f"| {FAULT_CLASSIFIER_LABELS[key]} | {m['accuracy']*100:.2f}% | {m['macro_f1']*100:.2f}% | "
            f"{m['train_time_seconds']:.3f} | {m['infer_time_ms_per_sample']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 四、RUL 剩余寿命预测归一化策略对比",
            "",
            "| 归一化策略 | 相对尺度 RMSE | 相对尺度 MAE | 相对尺度 R² | 分钟尺度 RMSE | 分钟尺度 MAE | 分钟尺度 R² |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for s_key in ("max_based", "group_minmax", "absolute"):
        item = rul[s_key]
        rs = item["relative_scale"]
        ms = item["minute_scale"]
        lines.append(
            f"| {item['label']} | {rs['rmse']:.4f} | {rs['mae']:.4f} | {rs['r2']:.4f} | "
            f"{ms['rmse']:.2f} | {ms['mae']:.2f} | {ms['r2']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 五、生成的对比图表",
            "",
            "- `reports/figures/bearing_model_comparison.png`：RF / 1D-CNN / LSTM 准确率、Macro-F1 与计算耗时对比",
            "- `reports/figures/bearing_confusion_matrices.png`：三模型混淆矩阵并排对比图",
            "- `reports/figures/bearing_accuracy_over_time.png`：三模型随退化时间推移准确率与 Macro-F1 曲线",
            "- `reports/figures/bearing_rul_degradation.png`：Max-based vs 分组归一化 vs 绝对 RUL 对比及退化曲线",
            "- `reports/figures/bearing_feature_correlation_pca.png`：Top-12 特征相关性热力图与 PCA 方差解释率",
        ]
    )
    return "\n".join(lines) + "\n"


def run_bearing_benchmark(
    data_dir: str | Path,
    report_dir: str | Path = PROJECT_ROOT / "reports",
    artifact_dir: str | Path = PROJECT_ROOT / "artifacts" / "models",
    epochs: int = 35,
    batch_size: int = 32,
    seed: int = 42,
    step: int = 1,
    save_agent_models: bool = True,
) -> dict[str, Any]:
    """执行西交大轴承多模型基准对比并生成报告与高清图表。"""
    cfg = load_config()
    loader = XJTUSYDataLoader(data_dir=data_dir, sampling_rate=cfg.sampling_rate)
    if not loader.is_available:
        raise RuntimeError(f"数据目录 `{data_dir}` 下未找到任何 XJTU-SY 轴承 CSV 数据。")

    report_root = Path(report_dir)
    figures_dir = report_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. 提取全量轴承 54 维时频特征
    all_bearings = loader.list_bearings()
    full_df = build_feature_dataset(loader, bearings=all_bearings, step=step)

    # 2. 特征相关性分析与 PCA 降维
    corr_report = compute_feature_correlation(full_df, feature_columns=FEATURE_COLUMNS)
    train_bearings, test_bearings = loader.split_bearings_by_unit(test_ratio=0.30, seed=seed)
    train_ids = {b.bearing_id for b in train_bearings}
    test_ids = {b.bearing_id for b in test_bearings}

    sp_train_full = full_df[full_df["bearing_id"].isin(train_ids)].reset_index(drop=True)
    sp_test_full = full_df[full_df["bearing_id"].isin(test_ids)].reset_index(drop=True)
    if sp_test_full.empty:
        sp_test_full = sp_train_full.copy()

    pca_reducer = PCADimensionalityReducer(
        feature_columns=FEATURE_COLUMNS, n_components=0.95, random_state=seed
    ).fit(sp_train_full)
    pca_summary = pca_reducer.explained_variance_summary()

    # 3. 空间 7:3 划分（退化阶段 [0.30, 0.90] 聚焦四类故障区分）
    sp_clf_train = sp_train_full[
        (sp_train_full["life_ratio"] >= 0.30) & (sp_train_full["life_ratio"] <= 0.90)
    ].reset_index(drop=True)
    sp_clf_test = sp_test_full[
        (sp_test_full["life_ratio"] >= 0.30) & (sp_test_full["life_ratio"] <= 0.90)
    ].reset_index(drop=True)

    sp_rf = RandomForestFaultClassifier(
        feature_columns=FEATURE_COLUMNS, n_estimators=200, random_state=seed
    ).fit(sp_clf_train, sp_clf_train["fault_type"])
    sp_cnn = CNN1DFaultClassifier(
        feature_columns=FEATURE_COLUMNS, epochs=epochs, batch_size=batch_size, random_state=seed
    ).fit(sp_clf_train, sp_clf_train["fault_type"])
    sp_lstm = LSTMFaultClassifier(
        feature_columns=FEATURE_COLUMNS, epochs=epochs, batch_size=batch_size, random_state=seed
    ).fit(sp_clf_train, sp_clf_train["fault_type"])

    spatial_models = {
        "rf": sp_rf.evaluate(sp_clf_test, sp_clf_test["fault_type"]),
        "cnn1d": sp_cnn.evaluate(sp_clf_test, sp_clf_test["fault_type"]),
        "lstm": sp_lstm.evaluate(sp_clf_test, sp_clf_test["fault_type"]),
    }

    # PCA 降维后 RF 效果对比
    sp_train_pca = pca_reducer.transform(sp_clf_train)
    sp_test_pca = pca_reducer.transform(sp_clf_test)
    rf_pca = RandomForestFaultClassifier(
        feature_columns=pca_reducer.output_columns, n_estimators=200, random_state=seed
    ).fit(sp_train_pca, sp_clf_train["fault_type"])
    rf_pca_eval = rf_pca.evaluate(sp_test_pca, sp_clf_test["fault_type"])

    # 4. 时间窗划分（训练集 20%~55%，测试集 55%~70%）+ 随时间推移性能变化趋势
    tw_train, tw_test = split_dataframe_by_time_window(
        full_df, train_window=(0.20, 0.55), test_window=(0.55, 0.70)
    )
    tw_rf = RandomForestFaultClassifier(
        feature_columns=FEATURE_COLUMNS, n_estimators=200, random_state=seed
    ).fit(tw_train, tw_train["fault_type"])
    tw_cnn = CNN1DFaultClassifier(
        feature_columns=FEATURE_COLUMNS, epochs=epochs, batch_size=batch_size, random_state=seed
    ).fit(tw_train, tw_train["fault_type"])
    tw_lstm = LSTMFaultClassifier(
        feature_columns=FEATURE_COLUMNS, epochs=epochs, batch_size=batch_size, random_state=seed
    ).fit(tw_train, tw_train["fault_type"])

    time_window_models = {
        "rf": tw_rf.evaluate(tw_test, tw_test["fault_type"]),
        "cnn1d": tw_cnn.evaluate(tw_test, tw_test["fault_type"]),
        "lstm": tw_lstm.evaluate(tw_test, tw_test["fault_type"]),
    }

    # 在全生命周期 [20%~100%] 各时间分箱上追踪时间窗训练模型的泛化随时间变化趋势
    eval_over_time_df = full_df[full_df["life_ratio"] >= 0.20].reset_index(drop=True)
    time_curves = {
        "rf": evaluate_predictions_over_time_bins(
            eval_over_time_df, eval_over_time_df["fault_type"], tw_rf.predict(eval_over_time_df)
        ),
        "cnn1d": evaluate_predictions_over_time_bins(
            eval_over_time_df, eval_over_time_df["fault_type"], tw_cnn.predict(eval_over_time_df)
        ),
        "lstm": evaluate_predictions_over_time_bins(
            eval_over_time_df, eval_over_time_df["fault_type"], tw_lstm.predict(eval_over_time_df)
        ),
    }

    # 5. RUL 剩余寿命预测归一化策略对比（Max-based vs 工况分组 vs 绝对分钟）
    rul_comparison_full = compare_rul_normalization_strategies(
        train_df=sp_train_full,
        test_df=sp_test_full,
        feature_columns=FEATURE_COLUMNS,
        n_estimators=200,
        random_state=seed,
    )

    # 6. 生成 5 张高清 PNG 对比图
    fig_paths = {
        "model_comparison": str(
            _plot_model_comparison(
                spatial_models, time_window_models, figures_dir / "bearing_model_comparison.png"
            )
        ),
        "confusion_matrices": str(
            _plot_confusion_matrices(spatial_models, figures_dir / "bearing_confusion_matrices.png")
        ),
        "accuracy_over_time": str(
            _plot_accuracy_over_time(time_curves, figures_dir / "bearing_accuracy_over_time.png")
        ),
        "rul_degradation": str(
            _plot_rul_degradation(
                sp_test_full, rul_comparison_full, figures_dir / "bearing_rul_degradation.png"
            )
        ),
        "feature_correlation_pca": str(
            _plot_feature_correlation_and_pca(
                corr_report, pca_summary, figures_dir / "bearing_feature_correlation_pca.png"
            )
        ),
    }

    # 7. 可选：同步训练并保存带“正常”类别的部署版 RF / 1D-CNN / LSTM 模型到 artifacts/models/
    if save_agent_models:
        pipeline = TrainingPipeline(
            loader=loader,
            artifact_dir=artifact_dir,
            step=step,
            random_state=seed,
            train_deep_models=True,
            deep_epochs=epochs,
            deep_batch_size=batch_size,
        )
        pipeline.run()

    # 精简 RUL 预测数组后保存 JSON 报告
    rul_summary = {
        k: {key: val for key, val in v.items() if not key.startswith("predictions_")}
        for k, v in rul_comparison_full.items()
    }

    report: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(Path(data_dir).resolve()),
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "step": step,
        },
        "dataset_summary": {
            "total_bearings": len(all_bearings),
            "conditions": loader.list_conditions(),
            "total_samples": int(len(full_df)),
            "feature_dim": len(FEATURE_COLUMNS),
        },
        "feature_engineering": {
            "correlation": corr_report.to_summary_dict(),
            "pca": pca_summary,
            "rf_with_pca": {
                "accuracy": rf_pca_eval["accuracy"],
                "macro_f1": rf_pca_eval["macro_f1"],
                "train_time_seconds": rf_pca_eval["train_time_seconds"],
                "infer_time_ms_per_sample": rf_pca_eval["infer_time_ms_per_sample"],
            },
        },
        "spatial_split_70_30": {
            "train_bearings": [b.key for b in train_bearings],
            "test_bearings": [b.key for b in test_bearings],
            "train_samples": int(len(sp_clf_train)),
            "test_samples": int(len(sp_clf_test)),
            "models": spatial_models,
        },
        "time_window_split": {
            "train_window": [0.20, 0.55],
            "test_window": [0.55, 0.70],
            "train_samples": int(len(tw_train)),
            "test_samples": int(len(tw_test)),
            "models": time_window_models,
            "metrics_over_time": time_curves,
        },
        "rul_comparison": rul_summary,
        "figures": {k: Path(v).name for k, v in fig_paths.items()},
    }

    json_path = report_root / "bearing_benchmark_report.json"
    md_path = report_root / "bearing_benchmark_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown_report(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="西交大 XJTU-SY 轴承多模型对比与 RUL 归一化基准脚本")
    parser.add_argument("--data-dir", default=str(cfg.data_dir), help="XJTU-SY 数据目录")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "reports"), help="报告与图表输出目录")
    parser.add_argument("--artifact-dir", default=str(cfg.artifact_dir), help="部署模型工件目录")
    parser.add_argument("--generate-mock", action="store_true", help="无数据时自动生成模拟 XJTU-SY 数据")
    parser.add_argument("--files-per-bearing", type=int, default=24, help="模拟数据单轴承基准文件数")
    parser.add_argument("--samples-per-file", type=int, default=32768, help="模拟数据单文件采样点数")
    parser.add_argument("--epochs", type=int, default=35, help="1D-CNN / LSTM 训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="1D-CNN / LSTM 批大小")
    parser.add_argument("--step", type=int, default=1, help="采样步长（真实大数据集可增大加速）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no-save-models", action="store_true", help="仅生成对比报告，不覆盖部署模型目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    data_dir = Path(args.data_dir)
    loader = XJTUSYDataLoader(data_dir=data_dir, sampling_rate=cfg.sampling_rate)
    if not loader.is_available:
        if args.generate_mock:
            from generate_mock_data import generate_mock_xjtu_dataset

            print(f"[INFO] 数据目录 {data_dir} 为空，正在生成模拟 XJTU-SY 数据...")
            generate_mock_xjtu_dataset(
                output_dir=data_dir,
                files_per_bearing=args.files_per_bearing,
                samples_per_file=args.samples_per_file,
                sampling_rate=cfg.sampling_rate,
            )
        else:
            raise SystemExit(
                f"[错误] 未在 `{data_dir}` 找到 XJTU-SY 轴承数据，请放入真实数据或附加 `--generate-mock`。"
            )

    print("[INFO] 开始执行西交大轴承多模型基准对比实验（RF / 1D-CNN / LSTM + RUL 归一化）...")
    report = run_bearing_benchmark(
        data_dir=data_dir,
        report_dir=args.report_dir,
        artifact_dir=args.artifact_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        step=args.step,
        save_agent_models=not args.no_save_models,
    )

    print("\n================ 西交大轴承多模型对比结果摘要 ================")
    for split_label, block in (
        ("按轴承单元 7:3 空间划分", report["spatial_split_70_30"]["models"]),
        ("时间窗划分 (20%~55% -> 55%~70%)", report["time_window_split"]["models"]),
    ):
        print(f"\n[{split_label}]")
        for k in ("rf", "cnn1d", "lstm"):
            m = block[k]
            print(
                f"  - {FAULT_CLASSIFIER_LABELS[k]:18s}: "
                f"Acc={m['accuracy']*100:6.2f}% | Macro-F1={m['macro_f1']*100:6.2f}% | "
                f"Train={m['train_time_seconds']:.3f}s | Infer={m['infer_time_ms_per_sample']:.4f}ms/sample"
            )

    print("\n[RUL 剩余寿命归一化对比（统一 [0,1] 相对尺度）]")
    for s_key in ("max_based", "group_minmax", "absolute"):
        r = report["rul_comparison"][s_key]
        rs = r["relative_scale"]
        print(f"  - {r['label']:22s}: RMSE={rs['rmse']:.4f} | MAE={rs['mae']:.4f} | R²={rs['r2']:.4f}")

    print(f"\n[OK] 报告与图表已输出至：{Path(args.report_dir).resolve()}")


if __name__ == "__main__":
    main()
