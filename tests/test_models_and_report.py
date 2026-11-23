"""异常检测、故障分类、RUL 预测模型与诊断报告模块单元测试。"""

from __future__ import annotations

from pathlib import Path

from equipment_agent.config import AppConfig
from equipment_agent.features.extractor import extract_sample_features
from equipment_agent.models.trainer import ModelBundle
from equipment_agent.report.generator import DiagnosticReportGenerator


def test_models_trained_and_persisted(trained_env) -> None:
    loader, bundle, artifact_dir = trained_env

    # 验证持久化加载一致性
    assert ModelBundle.exists(artifact_dir)
    reloaded = ModelBundle.load(artifact_dir)

    m = reloaded.metrics
    # 孤立森林在晚期退化段应具备高检出率
    assert m["anomaly_detection"]["late_stage_detection_rate"] >= 0.85
    # 随机森林在模拟数据上的故障分类与 RUL 预测应表现良好
    assert m["fault_classification"]["accuracy"] >= 0.70
    assert m["rul_prediction"]["mae"] <= 0.25

    # 对比同一轴承早期与晚期预测行为
    info = loader.get_bearing("Bearing1_1")
    early_feats = extract_sample_features(loader.load_file("Bearing1_1", 1))
    late_feats = extract_sample_features(loader.load_file("Bearing1_1", info.total_life_minutes))

    early_anom = reloaded.anomaly_detector.detect_one(early_feats)
    late_anom = reloaded.anomaly_detector.detect_one(late_feats)
    assert late_anom.anomaly_score > early_anom.anomaly_score
    assert late_anom.is_anomaly is True

    early_rul = reloaded.rul_predictor.predict_one(early_feats, condition=info.condition)
    late_rul = reloaded.rul_predictor.predict_one(late_feats, condition=info.condition)
    assert early_rul.rul_ratio > late_rul.rul_ratio


def test_rule_template_diagnostic_report(trained_env) -> None:
    loader, bundle, _ = trained_env
    info = loader.get_bearing("Bearing2_1")
    sample = loader.load_file("Bearing2_1", info.total_life_minutes)
    feats = extract_sample_features(sample)

    anom = bundle.anomaly_detector.detect_one(feats).to_dict()
    clf = bundle.fault_classifier.classify_one(feats).to_dict()
    rul = bundle.rul_predictor.predict_one(
        feats, condition=info.condition, reference_total_life=float(info.total_life_minutes)
    ).to_dict()

    # 无 API Key 配置下应自动退回规则模板引擎
    offline_cfg = AppConfig(openai_api_key="")
    generator = DiagnosticReportGenerator(config=offline_cfg)
    report = generator.generate(
        bearing_id=sample.bearing_id,
        condition=sample.condition,
        file_index=sample.file_index,
        anomaly_result=anom,
        classification_result=clf,
        rul_result=rul,
        features=feats,
        prefer_llm=True,
    )

    assert report.source == "rule_template"
    assert report.severity in {"正常", "轻度", "中度", "严重", "危急"}
    assert len(report.recommended_actions) >= 2
    assert len(report.inspection_checklist) >= 1

    md = report.to_markdown()
    assert "设备运维诊断报告" in md
    assert "Bearing2_1" in md
    assert "运维处置建议" in md
