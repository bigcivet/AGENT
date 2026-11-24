"""数据分析与多模型对比看板 — Streamlit 工作台第 3 页面。

展示：
1. 西交大 XJTU-SY 滚动轴承数据分析与多模型对比（RF / 1D-CNN / LSTM 空间 7:3 划分、
   时间窗 20%~55% → 55%~70% 划分、随时间推移准确率曲线、PCA 降维与相关性分析、
   Max-based vs 工况分组 vs 绝对 RUL 归一化对比）；
2. NEU-DET 金属表面缺陷多模型对比（70%/15%/15% 三划分下 ResNet18 vs EfficientNet-B0
   的测试集 Accuracy / Macro-F1、每类 Precision/Recall/F1、推理耗时与 6×6 混淆矩阵）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from equipment_agent.models.deep_classifiers import FAULT_CLASSIFIER_LABELS
from vision_agent.config import DEFECT_CLASSES, DEFECT_CN_MAP
from vision_agent.vision.classifier import VISION_MODEL_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _render_figure(fig_path: Path, caption: str) -> None:
    if fig_path.exists():
        st.image(str(fig_path), caption=caption, use_container_width=True)
    else:
        st.info(f"图表文件尚未生成：`{fig_path.name}`")


def _render_bearing_tab(report_dir: Path) -> None:
    st.subheader("⚙️ 西交大 XJTU-SY 轴承数据分析与多模型对比")
    report_path = report_dir / "bearing_benchmark_report.json"
    report = _load_json(report_path)

    if report is None:
        st.warning(
            "尚未检测到轴承多模型对比报告（`reports/bearing_benchmark_report.json`）。\n\n"
            "请在终端执行一键基准脚本生成真实实验数据与图表：\n"
            "```bash\npython scripts/run_bearing_benchmark.py --generate-mock\n```"
        )
        return

    figures_dir = report_dir / "figures"
    sp = report["spatial_split_70_30"]
    tw = report["time_window_split"]
    rul = report["rul_comparison"]
    fe = report["feature_engineering"]
    ds = report["dataset_summary"]

    st.caption(
        f"生成时间：`{report.get('generated_at', '-')}` | "
        f"轴承数：`{ds['total_bearings']}` 个 | 总样本：`{ds['total_samples']}` 条 | "
        f"原始特征：`{ds['feature_dim']}` 维 | "
        f"深度学习配置：`epochs={report['config']['epochs']}, batch_size={report['config']['batch_size']}`"
    )

    # 顶层 KPI
    best_sp_key = max(("rf", "cnn1d", "lstm"), key=lambda k: sp["models"][k]["accuracy"])
    best_tw_key = max(("rf", "cnn1d", "lstm"), key=lambda k: tw["models"][k]["accuracy"])
    max_rul_rel = rul["max_based"]["relative_scale"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "空间 7:3 最高准确率",
        f"{sp['models'][best_sp_key]['accuracy']*100:.2f}%",
        FAULT_CLASSIFIER_LABELS[best_sp_key],
    )
    k2.metric(
        "时间窗测试最高准确率",
        f"{tw['models'][best_tw_key]['accuracy']*100:.2f}%",
        FAULT_CLASSIFIER_LABELS[best_tw_key],
    )
    k3.metric(
        "Max-based RUL R²",
        f"{max_rul_rel['r2']:.4f}",
        f"RMSE={max_rul_rel['rmse']:.4f} / MAE={max_rul_rel['mae']:.4f}",
    )
    k4.metric(
        "PCA 降维压缩 (95%方差)",
        f"{fe['pca']['input_dim']} → {fe['pca']['output_dim']} 维",
        f"RF Acc={fe['rf_with_pca']['accuracy']*100:.1f}%",
    )

    # 1. 空间划分与时间窗划分多模型对比表
    st.markdown("#### 1. 故障分类多模型基准对比（RF vs 1D-CNN vs LSTM）")
    c_sp, c_tw = st.columns(2)
    with c_sp:
        st.markdown(
            f"**按轴承单元 7:3 空间划分**（训练 {sp['train_samples']} / 测试 {sp['test_samples']} 条）"
        )
        sp_rows = [
            {
                "模型": FAULT_CLASSIFIER_LABELS[k],
                "Accuracy": f"{sp['models'][k]['accuracy']*100:.2f}%",
                "Macro-F1": f"{sp['models'][k]['macro_f1']*100:.2f}%",
                "训练耗时(s)": sp["models"][k]["train_time_seconds"],
                "推理耗时(ms/样本)": sp["models"][k]["infer_time_ms_per_sample"],
            }
            for k in ("rf", "cnn1d", "lstm")
        ]
        st.dataframe(pd.DataFrame(sp_rows), use_container_width=True, hide_index=True)

    with c_tw:
        st.markdown(
            f"**时间窗划分（20%~55% → 55%~70%）**（训练 {tw['train_samples']} / 测试 {tw['test_samples']} 条）"
        )
        tw_rows = [
            {
                "模型": FAULT_CLASSIFIER_LABELS[k],
                "Accuracy": f"{tw['models'][k]['accuracy']*100:.2f}%",
                "Macro-F1": f"{tw['models'][k]['macro_f1']*100:.2f}%",
                "训练耗时(s)": tw["models"][k]["train_time_seconds"],
                "推理耗时(ms/样本)": tw["models"][k]["infer_time_ms_per_sample"],
            }
            for k in ("rf", "cnn1d", "lstm")
        ]
        st.dataframe(pd.DataFrame(tw_rows), use_container_width=True, hide_index=True)

    _render_figure(
        figures_dir / "bearing_model_comparison.png",
        "RF / 1D-CNN / LSTM 空间划分、时间窗划分与计算开销对比图",
    )
    _render_figure(
        figures_dir / "bearing_confusion_matrices.png",
        "RF / 1D-CNN / LSTM 故障分类混淆矩阵并排对比图",
    )
    _render_figure(
        figures_dir / "bearing_accuracy_over_time.png",
        "三模型在轴承生命周期不同时间区间的准确率与 Macro-F1 变化趋势曲线",
    )

    # 2. RUL 归一化对比
    st.markdown("---")
    st.markdown("#### 2. RUL 剩余寿命预测归一化策略对比（Max-based vs 分组归一化 vs 绝对分钟）")
    rul_rows = []
    for key in ("max_based", "group_minmax", "absolute"):
        item = rul[key]
        rs = item["relative_scale"]
        ms = item["minute_scale"]
        rul_rows.append(
            {
                "归一化策略": item["label"],
                "相对尺度 RMSE": rs["rmse"],
                "相对尺度 MAE": rs["mae"],
                "相对尺度 R²": rs["r2"],
                "分钟尺度 RMSE": ms["rmse"],
                "分钟尺度 MAE": ms["mae"],
                "分钟尺度 R²": ms["r2"],
            }
        )
    st.dataframe(pd.DataFrame(rul_rows), use_container_width=True, hide_index=True)
    _render_figure(
        figures_dir / "bearing_rul_degradation.png",
        "RUL 归一化策略误差对比与典型测试轴承退化预测曲线",
    )

    # 3. 特征相关性与 PCA 降维
    st.markdown("---")
    st.markdown("#### 3. 54 维时频特征相关性分析与 PCA 降维")
    top_rul_feats = fe["correlation"].get("top_rul_correlated_features", [])
    if top_rul_feats:
        col_t, col_fig = st.columns([1.0, 1.8])
        with col_t:
            st.markdown("**与剩余寿命相关性 Top-10 特征**")
            fdf = pd.DataFrame(top_rul_feats[:10]).rename(
                columns={"feature": "特征名", "corr_with_rul": "皮尔逊 r", "abs_corr": "|r|"}
            )
            st.dataframe(fdf, use_container_width=True, hide_index=True)
        with col_fig:
            _render_figure(
                figures_dir / "bearing_feature_correlation_pca.png",
                "Top-12 退化敏感特征相关性热力图与 PCA 累计方差解释率",
            )


def _render_neu_tab(report_dir: Path) -> None:
    st.subheader("🔍 NEU-DET 金属表面瑕疵多模型分类对比（ResNet18 vs EfficientNet-B0）")
    report_path = report_dir / "neu_benchmark_report.json"
    report = _load_json(report_path)

    if report is None:
        st.warning(
            "尚未检测到 NEU-DET 多模型对比报告（`reports/neu_benchmark_report.json`）。\n\n"
            "请在终端执行一键基准脚本生成真实实验数据与图表：\n"
            "```bash\npython scripts/run_neu_benchmark.py --generate-mock --epochs 5 --freeze-backbone\n```"
        )
        return

    figures_dir = report_dir / "figures"
    sp = report["split_summary"]
    models_m = report["models"]
    r18 = models_m["resnet18"]
    eff = models_m["efficientnet_b0"]

    st.caption(
        f"生成时间：`{report.get('generated_at', '-')}` | "
        f"70/15/15 分层划分：总计 `{sp['total_images']}` 张（Train `{sp['train_images']}` / "
        f"Val `{sp['val_images']}` / Test `{sp['test_images']}`） | "
        f"Epochs=`{report['config']['epochs']}`，输入尺寸=`{report['config']['image_size']}`"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ResNet18 测试准确率", f"{r18['accuracy']*100:.2f}%", f"Macro-F1 {r18['macro_f1']*100:.2f}%")
    k2.metric(
        "EfficientNet-B0 测试准确率",
        f"{eff['accuracy']*100:.2f}%",
        f"Macro-F1 {eff['macro_f1']*100:.2f}%",
    )
    k3.metric("ResNet18 单图推理", f"{r18['avg_infer_time_ms']:.2f} ms", f"参数量 {r18['total_params']/1e6:.2f}M")
    k4.metric(
        "EfficientNet-B0 单图推理",
        f"{eff['avg_infer_time_ms']:.2f} ms",
        f"参数量 {eff['total_params']/1e6:.2f}M",
    )

    # 总体对比表
    st.markdown("#### 1. 总体指标与计算效率对比（独立测试集）")
    summary_rows = [
        {
            "模型": VISION_MODEL_LABELS[k],
            "预训练权重": "ImageNet 预训练" if models_m[k]["pretrained_weights_used"] else "随机初始化(离线回退)",
            "参数量 (M)": round(models_m[k]["total_params"] / 1e6, 2),
            "Test Accuracy": f"{models_m[k]['accuracy']*100:.2f}%",
            "Test Macro-F1": f"{models_m[k]['macro_f1']*100:.2f}%",
            "训练耗时 (s)": models_m[k]["train_time_seconds"],
            "单图推理耗时 (ms)": models_m[k]["avg_infer_time_ms"],
        }
        for k in ("resnet18", "efficientnet_b0")
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # 逐类 Precision / Recall / F1 明细表
    st.markdown("#### 2. 6 类表面缺陷逐类 Precision / Recall / F1 对比")
    per_rows = []
    for cls in DEFECT_CLASSES:
        rm = r18["per_class_metrics"][cls]
        em = eff["per_class_metrics"][cls]
        per_rows.append(
            {
                "缺陷类别": f"{DEFECT_CN_MAP.get(cls, cls)} ({cls})",
                "ResNet18 P": rm["precision"],
                "ResNet18 R": rm["recall"],
                "ResNet18 F1": rm["f1_score"],
                "EfficientNet-B0 P": em["precision"],
                "EfficientNet-B0 R": em["recall"],
                "EfficientNet-B0 F1": em["f1_score"],
                "测试样本数": rm["support"],
            }
        )
    st.dataframe(pd.DataFrame(per_rows), use_container_width=True, hide_index=True)

    # 图表展示
    st.markdown("#### 3. 模型性能与混淆矩阵可视化对比")
    _render_figure(
        figures_dir / "neu_model_comparison.png",
        "ResNet18 vs EfficientNet-B0 准确率、Macro-F1、逐类 F1 与推理耗时对比",
    )
    _render_figure(
        figures_dir / "neu_confusion_matrices.png",
        "ResNet18 vs EfficientNet-B0 独立测试集 6×6 混淆矩阵对比",
    )


def render_benchmark_workbench() -> None:
    """渲染工作台第 3 页面：数据分析与模型对比。"""
    st.title("📊 工业数据分析与多模型对比中心")
    st.caption(
        "统一汇总展示西交大 XJTU-SY 滚动轴承（RF / 1D-CNN / LSTM + RUL 归一化）与 "
        "东北大学 NEU-DET 表面缺陷（ResNet18 vs EfficientNet-B0）多模型基准对比实验结果。"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 对比报告目录")
    report_dir_str = st.sidebar.text_input("Benchmark 报告目录", value=str(DEFAULT_REPORT_DIR))
    report_dir = Path(report_dir_str)

    tab_bearing, tab_neu = st.tabs(
        [
            "⚙️ 西交大轴承多模型与 RUL 对比（XJTU-SY）",
            "🔍 金属表面缺陷多模型对比（NEU-DET）",
        ]
    )
    with tab_bearing:
        _render_bearing_tab(report_dir)
    with tab_neu:
        _render_neu_tab(report_dir)
