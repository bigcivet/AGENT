"""设备运维诊断智能体 — Streamlit 交互式运维工作台。

启动命令：
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# 保证无论在哪个目录启动，都能正确导入 src/equipment_agent、src/vision_agent 与 app/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
APP_DIR = PROJECT_ROOT / "app"
for p in (SRC_DIR, SCRIPTS_DIR, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from equipment_agent.agent.orchestrator import EquipmentDiagnosisAgent
from equipment_agent.agent.tools import EquipmentToolkit
from equipment_agent.config import load_config
from equipment_agent.data.loader import XJTUSYDataLoader
from equipment_agent.models.deep_classifiers import FAULT_CLASSIFIER_LABELS
from equipment_agent.models.trainer import ModelBundle, TrainingPipeline
from benchmark_ui import render_benchmark_workbench
from vision_ui import render_vision_workbench

# 配置中文字体支持（兼容 Windows 微软雅黑 / 黑体 / Linux 常见字体）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="工业智能体综合工作台 · 设备运维 & 视觉质检",
    page_icon="🏭",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_resources(data_dir_str: str, artifact_dir_str: str, sampling_rate: int):
    """缓存数据加载器与模型束，避免每次交互重复磁盘 I/O。"""
    loader = XJTUSYDataLoader(data_dir=data_dir_str, sampling_rate=sampling_rate)
    bundle = ModelBundle.load(artifact_dir_str) if ModelBundle.exists(artifact_dir_str) else None
    return loader, bundle


def render_sidebar(cfg):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ 设备运维数据与模型")

    data_dir_str = st.sidebar.text_input("XJTU-SY 数据目录", value=str(cfg.data_dir))
    artifact_dir_str = st.sidebar.text_input("模型工件目录", value=str(cfg.artifact_dir))

    loader, bundle = load_resources(data_dir_str, artifact_dir_str, cfg.sampling_rate)

    # 2. 数据与模型一键准备
    st.sidebar.write(f"已扫描轴承数：**{len(loader.list_bearings())}**")
    st.sidebar.write(f"模型状态：{'✅ 已就绪' if bundle is not None else '⚠️ 未训练'}")

    selected_clf_key = "rf"
    if bundle is not None:
        avail_keys = bundle.available_classifiers()
        selected_clf_key = st.sidebar.selectbox(
            "故障分类模型切换（RF / 1D-CNN / LSTM）",
            options=avail_keys,
            format_func=lambda k: FAULT_CLASSIFIER_LABELS.get(k, k),
            index=0,
        )
        bundle.set_active_classifier(selected_clf_key)
        st.sidebar.caption(
            f"已就绪分类器：{', '.join(FAULT_CLASSIFIER_LABELS.get(k, k) for k in avail_keys)}"
        )

    train_deep_opt = st.sidebar.checkbox(
        "训练时同步包含 1D-CNN 与 LSTM",
        value=True,
        help="开启后点击“一键训练模型”将同时训练 RF / 1D-CNN / LSTM 三个分类器，便于实时切换对比。",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("生成模拟数据", use_container_width=True):
            from generate_mock_data import generate_mock_xjtu_dataset

            with st.spinner("正在生成符合 XJTU-SY 格式的模拟振动数据..."):
                generate_mock_xjtu_dataset(
                    output_dir=data_dir_str,
                    files_per_bearing=24,
                    samples_per_file=cfg.expected_samples_per_file,
                    sampling_rate=cfg.sampling_rate,
                )
            st.cache_resource.clear()
            st.rerun()

    with col2:
        if st.button("一键训练模型", use_container_width=True, disabled=not loader.is_available):
            with st.spinner("正在提取时频特征并训练异常检测/故障分类/RUL模型..."):
                pipeline = TrainingPipeline(
                    loader=loader,
                    artifact_dir=artifact_dir_str,
                    train_deep_models=train_deep_opt,
                )
                pipeline.run()
            st.cache_resource.clear()
            st.rerun()

    return loader, bundle, data_dir_str, artifact_dir_str


def render_chat_tab(agent: EquipmentDiagnosisAgent, loader: XJTUSYDataLoader):
    st.subheader("💬 工人提问 → AI 老师傅自主诊断")
    st.caption("智能体可自主调度孤立森林异常检测、随机森林故障分类、RUL 剩余寿命预测与结构化报告生成工具。")

    bearings = loader.list_bearings()
    default_bearing = bearings[0].bearing_id if bearings else "Bearing1_1"
    default_min = max(1, int(round(bearings[0].total_life_minutes * 0.85))) if bearings else 20

    # 快捷场景提问按钮
    st.markdown("**快捷提问示例：**")
    q_cols = st.columns(4)
    quick_prompts = [
        ("🔍 巡检全部轴承", "请帮我巡检一下所有轴承，看看哪些有异常风险？"),
        (f"🛠️ 全面诊断 {default_bearing}", f"帮我全面诊断 {default_bearing} 第 {default_min} 分钟的状态并给维修建议"),
        (f"⏳ 查 {default_bearing} 剩余寿命", f"{default_bearing} 第 {default_min} 分钟还能撑多久？"),
        ("📋 查看可用设备", "当前有哪些可用的轴承数据？"),
    ]
    triggered_prompt = None
    for col, (label, prompt) in zip(q_cols, quick_prompts):
        with col:
            if st.button(label, use_container_width=True):
                triggered_prompt = prompt

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "你好！我是设备运维诊断智能体（AI 老师傅）。\n\n"
                    "你可以直接问我任意轴承的运行健康状态、故障类型、剩余寿命（RUL）或维修处置建议。"
                    "例如点击上方的快捷提问按钮开始体验。"
                ),
                "tool_calls": [],
            }
        ]

    # 展示历史对话
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                with st.expander(f"🔧 查看智能体工具调用轨迹（共 {len(msg['tool_calls'])} 步）"):
                    for idx, tc in enumerate(msg["tool_calls"], 1):
                        st.markdown(f"**步骤 {idx}：`{tc['tool_name']}`** — 参数：`{json.dumps(tc['arguments'], ensure_ascii=False)}`")
                        # 隐藏冗长的 markdown 文本，保持轨迹清爽
                        preview = {k: v for k, v in tc["result"].items() if k != "markdown"}
                        st.json(preview)

    user_input = st.chat_input("输入设备运维问题，例如：Bearing2_1 第 20 分钟有没有异常？")
    prompt_to_run = triggered_prompt or user_input

    if prompt_to_run:
        st.session_state.chat_messages.append({"role": "user", "content": prompt_to_run})
        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        with st.chat_message("assistant"):
            with st.spinner("AI 老师傅正在读取振动信号并调用诊断工具..."):
                resp = agent.chat(prompt_to_run)
            st.markdown(resp.reply)

            serialized_tools = [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                }
                for tc in resp.tool_calls
            ]
            if serialized_tools:
                with st.expander(f"🔧 查看智能体工具调用轨迹（共 {len(serialized_tools)} 步）"):
                    for idx, tc in enumerate(serialized_tools, 1):
                        st.markdown(f"**步骤 {idx}：`{tc['tool_name']}`** — 参数：`{json.dumps(tc['arguments'], ensure_ascii=False)}`")
                        preview = {k: v for k, v in tc["result"].items() if k != "markdown"}
                        st.json(preview)

            if resp.report:
                st.session_state.latest_report = resp.report

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": resp.reply,
                "tool_calls": serialized_tools,
            }
        )
        if triggered_prompt:
            st.rerun()


def render_monitoring_tab(toolkit: EquipmentToolkit, loader: XJTUSYDataLoader):
    st.subheader("📈 逐文件实时异常监测与时频波形分析")
    bearings = loader.list_bearings()
    bearing_options = [b.bearing_id for b in bearings]

    c1, c2 = st.columns([2, 3])
    with c1:
        selected_bearing = st.selectbox("选择监测轴承", options=bearing_options)
        info = loader.get_bearing(selected_bearing)
        st.caption(f"工况：`{info.condition}` | 采样文件总数：`{info.total_life_minutes}` 分钟 | 数据集参考标签：`{info.fault_type}`")
    with c2:
        selected_minute = st.slider(
            "选择当前时间点（第 N 分钟 CSV 文件）",
            min_value=1,
            max_value=info.total_life_minutes,
            value=max(1, int(round(info.total_life_minutes * 0.8))),
        )

    # 实时单点诊断 KPI 卡片
    anom = toolkit.detect_anomaly(selected_bearing, selected_minute)
    clf = toolkit.classify_fault(selected_bearing, selected_minute)
    rul = toolkit.predict_rul(selected_bearing, selected_minute)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("异常检测状态", "⚠️ 异常" if anom["is_anomaly"] else "✅ 正常", f"健康度 {anom['health_index']:.1f}/100")
    k2.metric(
        "疑似故障类型",
        clf["fault_type"],
        f"置信度 {clf['confidence']*100:.1f}% · {clf.get('model_name', 'RF')}",
    )
    k3.metric("预测剩余寿命比例", f"{rul['rul_ratio']*100:.1f}%", f"约 {rul['estimated_rul_minutes']:.1f} 分钟")
    k4.metric("水平 RMS / 峭度", f"{anom['key_features']['h_rms']:.3f}", f"峭度 {anom['key_features']['h_kurtosis']:.2f}")

    # 逐文件全生命周期退化与异常检测曲线
    with st.spinner("正在逐文件计算生命周期监测曲线..."):
        trend_df = toolkit.monitor_bearing_trend(selected_bearing, step=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    # 子图 1：RMS 与峭度趋势
    axes[0].plot(trend_df["file_index"], trend_df["h_rms"], label="水平 RMS", color="#1f77b4")
    axes[0].plot(trend_df["file_index"], trend_df["v_rms"], label="垂直 RMS", color="#ff7f0e")
    axes[0].axvline(selected_minute, color="red", linestyle="--", alpha=0.7)
    axes[0].set_title("振动 RMS 退化趋势")
    axes[0].set_xlabel("运行时间（分钟）")
    axes[0].set_ylabel("RMS")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    # 子图 2：孤立森林异常分数
    axes[1].plot(trend_df["file_index"], trend_df["anomaly_score"], color="#d62728", label="异常分数")
    axes[1].axhline(0.0, color="black", linestyle=":", label="异常阈值 (0)")
    axes[1].axvline(selected_minute, color="red", linestyle="--", alpha=0.7)
    axes[1].set_title("Isolation Forest 异常分数")
    axes[1].set_xlabel("运行时间（分钟）")
    axes[1].set_ylabel("Anomaly Score")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    # 子图 3：预测 RUL vs 真实 RUL
    axes[2].plot(trend_df["file_index"], trend_df["true_rul_ratio"] * 100, label="真实剩余比例", color="#2ca02c")
    axes[2].plot(trend_df["file_index"], trend_df["pred_rul_ratio"] * 100, label="预测剩余比例", color="#9467bd", linestyle="--")
    axes[2].axvline(selected_minute, color="red", linestyle="--", alpha=0.7)
    axes[2].set_title("剩余寿命 (RUL) 预测对比")
    axes[2].set_xlabel("运行时间（分钟）")
    axes[2].set_ylabel("剩余比例 (%)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 当前时间点的时域波形与 FFT 频谱
    st.markdown(f"#### 当前采样文件（第 `{selected_minute}` 分钟）时域波形与 FFT 频谱")
    wf = toolkit.get_waveform_and_spectrum(selected_bearing, selected_minute)
    sample = wf["sample"]
    show_points = min(2048, len(sample.horizontal))

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 3.6))
    axes2[0].plot(wf["time_axis"][:show_points], sample.horizontal[:show_points], lw=0.7, label="水平振动")
    axes2[0].plot(wf["time_axis"][:show_points], sample.vertical[:show_points], lw=0.7, alpha=0.75, label="垂直振动")
    axes2[0].set_title(f"时域振动波形（前 {show_points} 点）")
    axes2[0].set_xlabel("时间 (s)")
    axes2[0].set_ylabel("幅值")
    axes2[0].legend(fontsize=8)
    axes2[0].grid(alpha=0.25)

    axes2[1].plot(wf["h_freqs"], wf["h_amps"], lw=0.8, label="水平 FFT")
    axes2[1].plot(wf["v_freqs"], wf["v_amps"], lw=0.8, alpha=0.75, label="垂直 FFT")
    axes2[1].set_title("单边 FFT 频谱")
    axes2[1].set_xlabel("频率 (Hz)")
    axes2[1].set_ylabel("幅值")
    axes2[1].legend(fontsize=8)
    axes2[1].grid(alpha=0.25)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # 支持上传自定义双通道 CSV 即时诊断
    st.markdown("---")
    st.markdown("#### 📤 上传单个双通道振动 CSV 文件即时诊断")
    uploaded = st.file_uploader("上传符合 XJTU-SY 格式的 CSV 文件（水平/垂直两列）", type=["csv"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)
        try:
            diag = toolkit.diagnose_csv_file(tmp_path, bearing_label=uploaded.name)
            st.markdown(diag["markdown"])
            st.session_state.latest_report = diag["report"]
        finally:
            tmp_path.unlink(missing_ok=True)


def render_report_and_metrics_tab(toolkit: EquipmentToolkit, bundle: ModelBundle, loader: XJTUSYDataLoader):
    st.subheader("📋 结构化诊断报告与模型离线评估")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("#### 生成指定轴承的结构化运维报告")
        bearings = loader.list_bearings()
        b_col, m_col, btn_col = st.columns([2, 2, 1.5])
        with b_col:
            target_b = st.selectbox("目标轴承", [b.bearing_id for b in bearings], key="rep_bearing")
            b_info = loader.get_bearing(target_b)
        with m_col:
            target_m = st.number_input(
                "时间点（分钟）",
                min_value=1,
                max_value=b_info.total_life_minutes,
                value=max(1, int(round(b_info.total_life_minutes * 0.85))),
                step=1,
            )
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("生成诊断报告", use_container_width=True):
                with st.spinner("正在生成结构化诊断报告..."):
                    res = toolkit.generate_diagnostic_report(target_b, int(target_m))
                st.session_state.latest_report = res["report"]
                st.session_state.latest_report_md = res["markdown"]

        if "latest_report" in st.session_state and st.session_state.latest_report:
            rep = st.session_state.latest_report
            md_text = st.session_state.get("latest_report_md")
            if not md_text:
                # 若来自对话生成，重新渲染完整结构化展示
                res = toolkit.generate_diagnostic_report(
                    rep["bearing_id"], int(rep["file_index"]), prefer_llm=False
                )
                md_text = res["markdown"]
            st.markdown(md_text)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "⬇️ 下载 JSON 报告",
                    data=json.dumps(rep, ensure_ascii=False, indent=2),
                    file_name=f"report_{rep['bearing_id']}_min{rep['file_index']}.json",
                    mime="application/json",
                )
            with d2:
                st.download_button(
                    "⬇️ 下载 Markdown 报告",
                    data=md_text,
                    file_name=f"report_{rep['bearing_id']}_min{rep['file_index']}.md",
                    mime="text/markdown",
                )
        else:
            st.info("点击上方“生成诊断报告”按钮，或在对话页向 AI 老师傅发起诊断即可查看完整报告。")

    with col_right:
        st.markdown("#### 🧪 当前模型离线评估指标")
        metrics = bundle.metrics
        if not metrics:
            st.warning("暂无评估指标文件。")
            return

        st.write(f"训练时间：`{metrics.get('trained_at', '-')}`")
        st.write(
            f"训练样本：`{metrics.get('train_samples', 0)}` | 测试样本：`{metrics.get('test_samples', 0)}`"
        )

        anom_m = metrics.get("anomaly_detection", {})
        clf_m = metrics.get("fault_classification", {})
        rul_m = metrics.get("rul_prediction", {})

        st.markdown("**1. 孤立森林异常检测**")
        st.write(
            f"- 早期健康段误报率：`{anom_m.get('early_false_alarm_rate', 0)*100:.1f}%`\n"
            f"- 晚期退化段检出率：`{anom_m.get('late_stage_detection_rate', 0)*100:.1f}%`"
        )

        st.markdown("**2. 随机森林故障分类（基线）**")
        st.write(
            f"- 准确率 (Accuracy)：`{clf_m.get('accuracy', 0)*100:.2f}%`\n"
            f"- 宏平均 F1 (Macro-F1)：`{clf_m.get('macro_f1', 0)*100:.2f}%`"
        )

        top_feats = clf_m.get("top_features", [])
        if top_feats:
            st.markdown("**Top 重要振动特征：**")
            feat_df = pd.DataFrame(top_feats, columns=["特征名", "重要性"])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.markdown("**3. 随机森林 RUL 剩余寿命预测（基线）**")
        st.write(
            f"- MAE：`{rul_m.get('mae', 0):.4f}` | RMSE：`{rul_m.get('rmse', 0):.4f}` | R²：`{rul_m.get('r2', 0):.4f}`"
        )


def render_equipment_workbench(cfg):
    st.title("🛠️ 工业设备运维诊断智能体（AI 老师傅）")
    st.caption("基于西交大 XJTU-SY 滚动轴承振动数据 · 时频特征工程 + 孤立森林异常检测 + 随机森林故障分类/RUL预测 + 大模型结构化诊断")

    loader, bundle, _, _ = render_sidebar(cfg)

    if not loader.is_available:
        st.warning(
            "当前数据目录下未检测到 XJTU-SY 轴承数据。请点击左侧栏的 **【生成模拟数据】** 按钮快速体验，"
            "或按 README 指引将真实 XJTU-SY 数据集放入 `data/xjtu_sy/` 目录。"
        )
        return

    if bundle is None:
        st.warning("已扫描到振动数据，但尚未训练诊断模型。请点击左侧栏的 **【一键训练模型】** 按钮。")
        return

    toolkit = EquipmentToolkit(loader=loader, bundle=bundle, config=cfg)
    if "agent" not in st.session_state:
        st.session_state.agent = EquipmentDiagnosisAgent(toolkit=toolkit, config=cfg)
    else:
        # 始终刷新底层 toolkit 引用
        st.session_state.agent.toolkit = toolkit

    tab_chat, tab_monitor, tab_report = st.tabs(
        ["💬 AI 老师傅对话诊断", "📈 实时监测与时频分析", "📋 结构化报告与模型评估"]
    )
    with tab_chat:
        render_chat_tab(st.session_state.agent, loader)
    with tab_monitor:
        render_monitoring_tab(toolkit, loader)
    with tab_report:
        render_report_and_metrics_tab(toolkit, bundle, loader)


def main():
    cfg = load_config()
    st.sidebar.title("🏭 工业智能体工作台")

    # 共用大模型连接状态展示
    if cfg.has_llm_key:
        st.sidebar.success(f"大模型模式已启用\n\n模型：`{cfg.openai_model}`")
    else:
        st.sidebar.info(
            "当前为 **本地规则回退模式**\n\n"
            "如需启用大模型（DeepSeek/通义千问等），请在 `.env` 中配置 `OPENAI_API_KEY`。"
        )

    module = st.sidebar.radio(
        "选择工作台页面",
        options=[
            "🔍 视觉质检智能体（NEU-DET）",
            "🛠️ 设备运维诊断智能体（XJTU-SY）",
            "📊 数据分析与模型对比",
        ],
        index=0,
        key="workbench_page_selector",
    )

    if module.startswith("🔍"):
        render_vision_workbench()
    elif module.startswith("🛠️"):
        render_equipment_workbench(cfg)
    else:
        render_benchmark_workbench()


if __name__ == "__main__":
    main()
