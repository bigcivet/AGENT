"""视觉质检智能体 — Streamlit 交互式界面模块。

作为统一工业智能体工作台的第二个页面，提供：
1. 📷 图片上传检测（支持上传/抽样检测、自动入库 SQLite、连续缺陷实时预警与工艺溯源）
2. 📊 质量数字孪生看板（KPI、各类缺陷分布、机台按天趋势、明细流水、ResNet18 评估与混淆矩阵）
3. 💬 质量对话查询（自然语言查库、受限只读 SQL、工具轨迹与工艺文件引用）
4. 🚨 主动预警与工艺知识库（阈值动态可配、预警闭环状态管理、本地工艺文档 RAG 检索）
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from vision_agent.agent.orchestrator import VisionQualityAgent
from vision_agent.agent.tools import VisionToolkit
from vision_agent.config import DEFECT_CLASSES, DEFECT_CN_MAP, VisionConfig, load_vision_config
from vision_agent.decision.alert_engine import AlertEngine
from vision_agent.decision.rag import ProcessDocRAG
from vision_agent.memory.store import QualityMemoryStore
from vision_agent.vision.classifier import ResNet18DefectClassifier
from vision_agent.vision.dataset import NEUDetCatalog
from vision_agent.vision.yolo_detector import YOLODefectDetector


@st.cache_resource(show_spinner=False)
def _load_vision_resources(
    neu_det_dir: str,
    artifact_dir: str,
    docs_dir: str,
):
    catalog = NEUDetCatalog(neu_det_dir)
    classifier = (
        ResNet18DefectClassifier.load(artifact_dir)
        if ResNet18DefectClassifier.exists(artifact_dir)
        else None
    )
    rag = ProcessDocRAG(docs_dir)
    return catalog, classifier, rag


def render_vision_sidebar(cfg: VisionConfig):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 视觉质检数据与模型")

    neu_dir = st.sidebar.text_input("NEU-DET 图像目录", value=str(cfg.neu_det_dir))
    art_dir = st.sidebar.text_input("视觉模型目录", value=str(cfg.vision_artifact_dir))
    db_path = st.sidebar.text_input("质量记忆库 (SQLite)", value=str(cfg.db_path))
    docs_dir = st.sidebar.text_input("工艺知识文档目录", value=str(cfg.process_docs_dir))

    alert_threshold = st.sidebar.slider(
        "连续同类缺陷预警阈值（次）",
        min_value=2,
        max_value=8,
        value=cfg.alert_consecutive_threshold,
        step=1,
    )

    catalog, classifier, rag = _load_vision_resources(neu_dir, art_dir, docs_dir)
    store = QualityMemoryStore(db_path)

    st.sidebar.write(f"已扫描缺陷图：**{len(catalog.samples)}** 张")
    st.sidebar.write(f"ResNet18 模型：{'✅ 已就绪' if classifier is not None else '⚠️ 未训练'}")
    st.sidebar.write(f"数字孪生流水：**{store.count_inspections()}** 条")
    st.sidebar.write(f"工艺文档数：**{len(rag.list_documents())}** 份（{len(rag.chunks)} 知识块）")
    yolo_flag = "✅ 已安装" if YOLODefectDetector.is_ultralytics_installed() else "可选（未安装 ultralytics）"
    st.sidebar.caption(f"YOLO 检测适配：{yolo_flag}")

    epochs = st.sidebar.number_input("ResNet18 训练轮数 (Epochs)", min_value=1, max_value=30, value=4, step=1)

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("生成模拟缺陷图", use_container_width=True):
            from generate_mock_neu_det import generate_mock_neu_det_dataset

            with st.spinner("正在生成 200x200 模拟 NEU-DET 缺陷图与 VOC XML..."):
                generate_mock_neu_det_dataset(output_dir=neu_dir, per_class=48)
            st.cache_resource.clear()
            st.rerun()

    with c2:
        if st.button("训练 ResNet18", use_container_width=True, disabled=not catalog.is_available):
            with st.spinner(f"正在训练 ResNet18 迁移学习模型（{epochs} 轮）..."):
                clf = ResNet18DefectClassifier(image_size=cfg.image_size)
                clf.fit(catalog=catalog, epochs=int(epochs), batch_size=16, freeze_backbone=True)
                clf.save(art_dir)
            st.cache_resource.clear()
            st.rerun()

    if st.sidebar.button("🌊 生成模拟多机台多天流水", use_container_width=True):
        from simulate_inspection_stream import simulate_inspection_stream

        with st.spinner("正在写入多机台 7 天模拟质检流水并扫描预警..."):
            simulate_inspection_stream(
                db_path=db_path,
                process_docs_dir=docs_dir,
                neu_det_dir=neu_dir,
                days=7,
                alert_threshold=int(alert_threshold),
                reset_db=True,
            )
        st.rerun()

    alert_engine = AlertEngine(store=store, rag=rag, consecutive_threshold=int(alert_threshold))
    toolkit = VisionToolkit(
        store=store,
        rag=rag,
        alert_engine=alert_engine,
        classifier=classifier,
        config=cfg,
    )
    return catalog, classifier, store, rag, alert_engine, toolkit


def _render_upload_inspect_tab(
    toolkit: VisionToolkit,
    catalog: NEUDetCatalog,
    classifier: ResNet18DefectClassifier | None,
):
    st.subheader("📷 上传金属表面图像检测与数字孪生入库")
    st.caption("支持上传任意带钢表面图片或一键抽取 NEU-DET 样本，识别后自动写入 SQLite 并触发连续缺陷主动预警。")

    if classifier is None:
        st.warning("尚未检测到已训练的 ResNet18 分类模型，请先点击左侧栏的 **【生成模拟缺陷图】** 和 **【训练 ResNet18】**。")
        return

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        machine_id = st.selectbox("检测机台号", options=["M01", "M02", "M03", "M04"], index=2)
    with col_cfg2:
        default_batch = f"B{datetime.now().strftime('%Y%m%d')}-{machine_id}-LIVE"
        batch_id = st.text_input("生产批次号", value=default_batch)
    with col_cfg3:
        notes = st.text_input("质检备注", value="人工上传复检")

    col_left, col_right = st.columns([1.2, 1.8])
    image_to_inspect: Image.Image | None = None
    image_label_path = ""

    with col_left:
        uploaded = st.file_uploader(
            "上传金属表面图像（jpg / png / bmp，建议 200×200 灰度图）",
            type=["jpg", "jpeg", "png", "bmp"],
        )
        if uploaded is not None:
            image_to_inspect = Image.open(uploaded)
            image_label_path = f"upload://{uploaded.name}"
        elif catalog.is_available:
            st.markdown("**或从本地 NEU-DET 数据集抽取样本测试：**")
            cls_choice = st.selectbox(
                "选择样本缺陷类别",
                options=list(DEFECT_CLASSES),
                format_func=lambda c: f"{DEFECT_CN_MAP.get(c, c)} ({c})",
            )
            candidates = [s for s in catalog.samples if s.label == cls_choice]
            if candidates:
                sample_idx = st.slider("样本序号", 0, len(candidates) - 1, 0)
                chosen = candidates[sample_idx]
                image_to_inspect = Image.open(chosen.image_path)
                image_label_path = str(chosen.image_path)

        if image_to_inspect is not None:
            st.image(image_to_inspect, caption=f"待检图像：{Path(image_label_path).name}", width=280)

    with col_right:
        if image_to_inspect is None:
            st.info("请在左侧上传一张金属表面图片或抽取数据集样本。")
            return

        if st.button("🚀 执行视觉识别并写入数字孪生库", type="primary", use_container_width=True):
            with st.spinner("正在调用 ResNet18 执行推理、写入 SQLite 并检索工艺知识..."):
                res = toolkit.inspect_image_and_record(
                    image_source=image_to_inspect,
                    machine_id=machine_id,
                    batch_id=batch_id,
                    image_path=image_label_path,
                    notes=notes,
                )
            st.session_state.last_vision_inspection = res

        res = st.session_state.get("last_vision_inspection")
        if res:
            pred = res["prediction"]
            k1, k2, k3 = st.columns(3)
            k1.metric("识别缺陷类型", f"{pred['defect_type_cn']}", pred["defect_type"])
            k2.metric("模型置信度", f"{pred['confidence'] * 100:.2f}%", f"记录 #{res['record_id']}")
            k3.metric("风险等级", pred["severity"], f"机台 {res['record']['machine_id']}")

            # 若触发连续缺陷主动预警，立即高亮提示
            alert = res.get("triggered_alert")
            if alert:
                st.error(
                    f"🚨 **触发主动预警 [{alert['severity']}]**：{alert['summary']}\n\n"
                    f"- **可能工艺原因**：{alert['possible_causes']}\n"
                    f"- **处置建议**：{alert['recommended_actions']}"
                )

            # 各类别置信度分布图
            probs = pred["probabilities"]
            prob_df = pd.DataFrame(
                [
                    {"缺陷类别": f"{DEFECT_CN_MAP.get(k, k)} ({k})", "概率": v}
                    for k, v in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
                ]
            )
            st.markdown("**6 类缺陷置信度分布：**")
            st.bar_chart(prob_df.set_index("缺陷类别"))

            guidance = res.get("process_guidance")
            if guidance:
                cites = "、".join(r["citation"] for r in guidance.get("references", [])[:2])
                with st.expander(f"📘 关联工艺成因与处置参考（{cites}）", expanded=True):
                    st.markdown(f"- **可能工艺原因**：{guidance['possible_causes']}")
                    st.markdown(f"- **推荐处置措施**：{guidance['recommended_actions']}")


def _render_dashboard_tab(
    store: QualityMemoryStore,
    classifier: ResNet18DefectClassifier | None,
):
    st.subheader("📊 产线质量数字孪生看板与模型评估")

    days = st.slider("统计时间窗口（天）", min_value=1, max_value=30, value=7)
    summary = store.get_defect_summary(days=days)
    machines = store.get_machine_comparison(days=days)
    open_alerts = store.list_alerts(status="open")

    if summary["total_records"] == 0:
        st.info("当前 SQLite 数字孪生库暂无流水记录。请点击左侧栏的 **【生成模拟多机台多天流水】** 或上传图片检测。")
        return

    top = summary["top_defect"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"近 {days} 天缺陷总数", f"{summary['total_records']} 条")
    k2.metric(
        "首要高发缺陷",
        f"{top['defect_type_cn']}" if top else "-",
        f"{top['count']} 次 ({top['ratio']*100:.1f}%)" if top else "",
    )
    k3.metric("监测机台数", f"{len(machines)} 台")
    k4.metric("活跃连续预警", f"{len(open_alerts)} 条")

    # 图表 1 & 2：缺陷类型分布 + 机台按天趋势
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### 各类表面缺陷分布")
        bd_df = pd.DataFrame(summary["breakdown"])
        fig1, ax1 = plt.subplots(figsize=(6.2, 3.6))
        labels = [f"{r['defect_type_cn']}\n({r['defect_type']})" for _, r in bd_df.iterrows()]
        bars = ax1.bar(labels, bd_df["count"], color="#2b6cb0")
        ax1.set_ylabel("检出次数")
        ax1.grid(axis="y", alpha=0.25)
        ax1.tick_params(axis="x", labelsize=8)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.3, str(int(h)), ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with c_right:
        st.markdown("#### 各机台每日缺陷趋势")
        trend_rows = store.get_daily_trend(days=days)
        if trend_rows:
            tdf = pd.DataFrame(trend_rows)
            pivot = tdf.pivot_table(index="date", columns="machine_id", values="count", aggfunc="sum", fill_value=0)
            fig2, ax2 = plt.subplots(figsize=(6.2, 3.6))
            for col in pivot.columns:
                ax2.plot(pivot.index, pivot[col], marker="o", linewidth=2, label=col)
            ax2.set_ylabel("当日缺陷数")
            ax2.tick_params(axis="x", rotation=25, labelsize=8)
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    # 机台横向对比表
    st.markdown("#### 机台质量横向对比")
    if machines:
        mdf = pd.DataFrame(machines).rename(
            columns={
                "machine_id": "机台号",
                "total_defects": "缺陷总数",
                "high_severity_count": "高危缺陷数",
                "avg_confidence": "平均置信度",
                "top_defect_cn": "主导缺陷",
                "top_defect_count": "主导缺陷次数",
            }
        )
        st.dataframe(
            mdf[["机台号", "缺陷总数", "高危缺陷数", "平均置信度", "主导缺陷", "主导缺陷次数"]],
            use_container_width=True,
            hide_index=True,
        )

    # 最近检测流水明细
    st.markdown("#### 最近检测流水明细（数字孪生档案）")
    f1, f2 = st.columns(2)
    with f1:
        m_filter = st.selectbox("按机台筛选", options=["全部"] + store.list_machines())
    with f2:
        d_filter = st.selectbox(
            "按缺陷筛选",
            options=["全部"] + list(DEFECT_CLASSES),
            format_func=lambda x: "全部" if x == "全部" else f"{DEFECT_CN_MAP.get(x, x)} ({x})",
        )
    recent = store.query_recent_inspections(
        limit=50,
        machine_id=None if m_filter == "全部" else m_filter,
        defect_type=None if d_filter == "全部" else d_filter,
    )
    if recent:
        rdf = pd.DataFrame(recent)[
            ["id", "inspected_at", "machine_id", "batch_id", "defect_type_cn", "defect_type", "confidence", "severity", "notes"]
        ]
        rdf.columns = ["ID", "检测时间", "机台", "批次", "缺陷(中)", "缺陷(英)", "置信度", "风险", "备注"]
        st.dataframe(rdf, use_container_width=True, hide_index=True)

    # ResNet18 离线评估与混淆矩阵
    if classifier is not None and classifier.metrics:
        st.markdown("---")
        st.markdown("#### 🧪 ResNet18 视觉分类模型评估与混淆矩阵")
        m = classifier.metrics
        mc1, mc2 = st.columns([1.2, 1.5])
        with mc1:
            st.write(f"- **训练时间**：`{m.get('trained_at', '-')}`")
            st.write(f"- **运行设备**：`{m.get('device', '-')}` | **预训练权重**：`{m.get('pretrained_weights_used')}`")
            st.write(f"- **训练/验证样本**：`{m.get('train_samples', 0)}` / `{m.get('val_samples', 0)}`")
            st.write(
                f"- **验证准确率 (Accuracy)**：**`{m.get('accuracy', 0)*100:.2f}%`** | "
                f"**Macro-F1**：**`{m.get('macro_f1', 0)*100:.2f}%`**"
            )
            per_cls = m.get("per_class_metrics", {})
            if per_cls:
                pdf = pd.DataFrame(
                    [
                        {
                            "类别": f"{v['cn_name']} ({k})",
                            "Precision": v["precision"],
                            "Recall": v["recall"],
                            "F1": v["f1_score"],
                            "样本数": v["support"],
                        }
                        for k, v in per_cls.items()
                    ]
                )
                st.dataframe(pdf, use_container_width=True, hide_index=True)
        with mc2:
            cm = np.array(m.get("confusion_matrix", []))
            if cm.size > 0:
                fig_cm, ax_cm = plt.subplots(figsize=(5.6, 4.4))
                im = ax_cm.imshow(cm, cmap="Blues")
                cls_labels = [DEFECT_CN_MAP.get(c, c) for c in m.get("classes", DEFECT_CLASSES)]
                ax_cm.set_xticks(range(len(cls_labels)))
                ax_cm.set_yticks(range(len(cls_labels)))
                ax_cm.set_xticklabels(cls_labels, rotation=30, ha="right", fontsize=8)
                ax_cm.set_yticklabels(cls_labels, fontsize=8)
                ax_cm.set_xlabel("预测类别")
                ax_cm.set_ylabel("真实类别")
                ax_cm.set_title("NEU-DET 验证集混淆矩阵")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax_cm.text(
                            j,
                            i,
                            str(int(cm[i, j])),
                            ha="center",
                            va="center",
                            color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                            fontsize=8,
                        )
                plt.colorbar(im, ax=ax_cm, fraction=0.046)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)


def _render_vision_chat_tab(agent: VisionQualityAgent):
    st.subheader("💬 产线质量自然语言查询（Tool Calling + 受限 SQL + 工艺 RAG）")
    st.caption("可直接用中文询问缺陷统计、机台异常趋势、连续预警或某类缺陷的工艺成因与处置措施。")

    quick_questions = [
        ("🏆 最近哪种缺陷最多？", "最近哪种缺陷最多？"),
        ("📈 3 号机台有异常趋势吗？", "3 号机台有没有异常趋势？"),
        ("🏭 各机台质量横向对比", "各机台近 7 天的缺陷数量对比情况如何？"),
        ("📘 压入氧化皮原因与处置", "压入氧化皮的常见工艺原因和处置措施是什么？"),
    ]
    cols = st.columns(4)
    triggered = None
    for col, (label, q) in zip(cols, quick_questions):
        with col:
            if st.button(label, use_container_width=True):
                triggered = q

    if "vision_chat_messages" not in st.session_state:
        st.session_state.vision_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "你好！我是金属表面视觉质检智能体。\n\n"
                    "我可以实时查询 SQLite 产线质量数字孪生记录、分析机台异常趋势，并结合 `data/process_docs/` 中的工艺规程为您溯源处置建议。"
                    "请点击上方快捷问题或直接输入提问。"
                ),
                "tool_calls": [],
            }
        ]

    for msg in st.session_state.vision_chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                with st.expander(f"🔧 查看工具调用与 RAG 检索轨迹（共 {len(msg['tool_calls'])} 步）"):
                    for idx, tc in enumerate(msg["tool_calls"], 1):
                        st.markdown(
                            f"**步骤 {idx}：`{tc['tool_name']}`** — 参数：`{json.dumps(tc['arguments'], ensure_ascii=False)}`"
                        )
                        st.json(tc["result"])

    user_input = st.chat_input("输入质量查询问题，例如：3 号机台有没有异常趋势？")
    prompt = triggered or user_input

    if prompt:
        st.session_state.vision_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在查询质量数字孪生数据库并检索工艺文件..."):
                resp = agent.chat(prompt)
            st.markdown(resp.reply)
            serialized = [
                {"tool_name": tc.tool_name, "arguments": tc.arguments, "result": tc.result}
                for tc in resp.tool_calls
            ]
            if serialized:
                with st.expander(f"🔧 查看工具调用与 RAG 检索轨迹（共 {len(serialized)} 步）"):
                    for idx, tc in enumerate(serialized, 1):
                        st.markdown(
                            f"**步骤 {idx}：`{tc['tool_name']}`** — 参数：`{json.dumps(tc['arguments'], ensure_ascii=False)}`"
                        )
                        st.json(tc["result"])

        st.session_state.vision_chat_messages.append(
            {"role": "assistant", "content": resp.reply, "tool_calls": serialized}
        )
        if triggered:
            st.rerun()


def _render_alerts_and_rag_tab(
    store: QualityMemoryStore,
    rag: ProcessDocRAG,
    alert_engine: AlertEngine,
):
    st.subheader("🚨 连续缺陷主动预警列表与工艺知识库（RAG）")

    c1, c2, c3 = st.columns([1.5, 1.5, 1.5])
    with c1:
        status_filter = st.selectbox("筛选预警状态", options=["open", "acknowledged", "resolved", "all"], index=0)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 按当前阈值重新扫描全库预警", use_container_width=True):
            triggered = alert_engine.scan_all_machines(recent_only=False)
            st.success(f"扫描完成，共发现/更新 {len(triggered)} 条满足阈值（连续 ≥ {alert_engine.consecutive_threshold} 次）的预警。")
            st.rerun()
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(f"当前预警阈值：同机台连续 **{alert_engine.consecutive_threshold}** 次同缺陷")

    alerts = store.list_alerts(status=status_filter)
    if not alerts:
        st.info("当前筛选条件下暂无预警记录。")
    else:
        for a in alerts:
            badge = {"紧急": "🔴", "严重": "🟠", "警告": "🟡"}.get(a["severity"], "🟡")
            header = (
                f"{badge} #{a['id']} [{a['severity']}] 机台 `{a['machine_id']}` — "
                f"连续 {a['consecutive_count']} 次「{a['defect_type_cn']}」 | 状态：{a['status']} | {a['created_at']}"
            )
            with st.expander(header, expanded=(a["status"] == "open")):
                st.markdown(f"**预警摘要**：{a['summary']}")
                st.markdown(f"**关联可能工艺原因**：{a['possible_causes']}")
                st.markdown(f"**推荐处置措施**：{a['recommended_actions']}")

                refs = a.get("doc_references", [])
                if refs:
                    st.markdown("**引用工艺文档片段：**")
                    for ref in refs:
                        st.markdown(f"- `{ref.get('citation', '')}`（相关度 `{ref.get('score', 0):.2f}`）")

                b1, b2, _ = st.columns([1, 1, 3])
                with b1:
                    if a["status"] == "open" and st.button("标记已确认", key=f"ack_{a['id']}"):
                        store.update_alert_status(a["id"], "acknowledged")
                        st.rerun()
                with b2:
                    if a["status"] != "resolved" and st.button("标记已闭环解决", key=f"res_{a['id']}"):
                        store.update_alert_status(a["id"], "resolved")
                        st.rerun()

    st.markdown("---")
    st.markdown("#### 📚 本地工艺文档 RAG 检索（`data/process_docs/`）")
    st.caption(f"已加载工艺文档：{', '.join(rag.list_documents()) or '无'}")
    rag_q = st.text_input("输入工艺关键词或缺陷名进行检索", value="划痕 导卫 辊道 处置措施")
    if rag_q.strip():
        hits = rag.search(rag_q, top_k=3)
        for idx, h in enumerate(hits, 1):
            with st.expander(f"命中 {idx}：{h.citation}（相关度 {h.score:.2f}）", expanded=(idx == 1)):
                st.markdown(h.content)


def render_vision_workbench():
    """视觉质检智能体完整工作台入口。"""
    cfg = load_vision_config()
    st.title("🔍 工业视觉质检智能体（NEU-DET 数字孪生质检台）")
    st.caption("PyTorch ResNet18 表面缺陷识别 + SQLite 产线质量数字孪生 + 自然语言查询 + 连续缺陷主动预警与工艺 RAG 溯源")

    catalog, classifier, store, rag, alert_engine, toolkit = render_vision_sidebar(cfg)

    if "vision_agent" not in st.session_state:
        st.session_state.vision_agent = VisionQualityAgent(toolkit=toolkit, config=cfg)
    else:
        st.session_state.vision_agent.toolkit = toolkit

    tab_upload, tab_board, tab_chat, tab_alerts = st.tabs(
        [
            "📷 图片上传检测",
            "📊 质量数字孪生看板",
            "💬 质量对话查询",
            "🚨 主动预警与工艺知识库",
        ]
    )
    with tab_upload:
        _render_upload_inspect_tab(toolkit, catalog, classifier)
    with tab_board:
        _render_dashboard_tab(store, classifier)
    with tab_chat:
        _render_vision_chat_tab(st.session_state.vision_agent)
    with tab_alerts:
        _render_alerts_and_rag_tab(store, rag, alert_engine)
