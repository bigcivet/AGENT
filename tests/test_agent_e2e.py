"""设备运维诊断智能体（Agent）工具调度与端到端对话集成测试。"""

from __future__ import annotations

from equipment_agent.agent.orchestrator import EquipmentDiagnosisAgent
from equipment_agent.agent.tools import EquipmentToolkit
from equipment_agent.config import AppConfig


def test_agent_full_diagnosis_and_multiturn_memory(trained_env) -> None:
    loader, bundle, _ = trained_env
    offline_cfg = AppConfig(openai_api_key="")
    toolkit = EquipmentToolkit(loader=loader, bundle=bundle, config=offline_cfg)
    agent = EquipmentDiagnosisAgent(toolkit=toolkit, config=offline_cfg)

    # 第 1 轮：列出可用设备
    r1 = agent.chat("目前有哪些可用的轴承？")
    assert r1.mode == "rule_router"
    assert [tc.tool_name for tc in r1.tool_calls] == ["list_available_bearings"]
    assert "Bearing1_1" in r1.reply

    # 第 2 轮：全面诊断指定轴承某时间点
    r2 = agent.chat("请帮我全面诊断 Bearing1_1 第 14 分钟的状态并给出维修建议")
    called = [tc.tool_name for tc in r2.tool_calls]
    assert "detect_anomaly" in called
    assert "classify_fault" in called
    assert "predict_rul" in called
    assert "generate_diagnostic_report" in called
    assert r2.report is not None
    assert r2.report["bearing_id"] == "Bearing1_1"
    assert r2.report["file_index"] == 14

    # 第 3 轮：上下文追问（未显式写轴承名，应记住 Bearing1_1）
    r3 = agent.chat("它大概还能撑多久？")
    assert [tc.tool_name for tc in r3.tool_calls] == ["predict_rul"]
    assert "Bearing1_1" in r3.reply

    # 第 4 轮：全量巡检
    r4 = agent.chat("巡检一下所有轴承，看看哪些有异常风险")
    assert [tc.tool_name for tc in r4.tool_calls] == ["scan_all_bearings_health"]
    assert "全量轴承巡检" in r4.reply


def test_toolkit_trend_and_csv_diagnosis(trained_env) -> None:
    loader, bundle, _ = trained_env
    offline_cfg = AppConfig(openai_api_key="")
    toolkit = EquipmentToolkit(loader=loader, bundle=bundle, config=offline_cfg)

    trend_df = toolkit.monitor_bearing_trend("Bearing1_1", step=2)
    assert not trend_df.empty
    assert {"file_index", "h_rms", "anomaly_score", "pred_rul_ratio"}.issubset(trend_df.columns)

    # 直接诊断单个原始 CSV 文件
    info = loader.get_bearing("Bearing1_1")
    diag = toolkit.diagnose_csv_file(info.files[-1], bearing_label="TestBearing", prefer_llm=False)
    assert diag["report"]["bearing_id"] == "TestBearing"
    assert "markdown" in diag
