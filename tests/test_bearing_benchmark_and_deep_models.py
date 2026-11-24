"""西交大轴承多模型（1D-CNN / LSTM / RF）、特征相关性与 PCA、时间窗划分及 RUL 归一化单元测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from equipment_agent.agent.tools import EquipmentToolkit
from equipment_agent.config import AppConfig
from equipment_agent.data.loader import (
    XJTUSYDataLoader,
    evaluate_predictions_over_time_bins,
    split_dataframe_by_time_window,
)
from equipment_agent.features.analysis import (
    PCADimensionalityReducer,
    compute_feature_correlation,
)
from equipment_agent.features.extractor import (
    FEATURE_COLUMNS,
    build_feature_dataset,
    extract_sample_features,
)
from equipment_agent.models.deep_classifiers import (
    CNN1DFaultClassifier,
    LSTMFaultClassifier,
    load_fault_classifier,
)
from equipment_agent.models.rul import compare_rul_normalization_strategies
from equipment_agent.models.trainer import ModelBundle, TrainingPipeline


def test_time_window_split_and_feature_analysis_pca(mock_dataset_dir: Path) -> None:
    loader = XJTUSYDataLoader(data_dir=mock_dataset_dir, sampling_rate=25600)
    df = build_feature_dataset(loader, step=1)
    assert not df.empty

    # 1. 时间窗划分（训练集 20%~55%，测试集 55%~70%）
    tw_train, tw_test = split_dataframe_by_time_window(
        df, train_window=(0.20, 0.55), test_window=(0.55, 0.70)
    )
    assert not tw_train.empty
    assert not tw_test.empty
    assert (tw_train["life_ratio"] >= 0.20).all()
    assert (tw_train["life_ratio"] < 0.55).all()
    assert (tw_test["life_ratio"] >= 0.55).all()
    assert (tw_test["life_ratio"] <= 0.70).all()

    # 2. 特征相关性分析
    corr_report = compute_feature_correlation(df, feature_columns=FEATURE_COLUMNS)
    assert corr_report.corr_matrix.shape == (len(FEATURE_COLUMNS), len(FEATURE_COLUMNS))
    assert len(corr_report.rul_correlations) == len(FEATURE_COLUMNS)
    # 退化物理特征（如 RMS/峰值等）应与 RUL 显著相关
    assert corr_report.rul_correlations[0]["abs_corr"] >= 0.5

    # 3. PCA 降维（保留 95% 累计方差）
    pca = PCADimensionalityReducer(feature_columns=FEATURE_COLUMNS, n_components=0.95, random_state=42)
    reduced = pca.fit_transform(tw_train)
    summary = pca.explained_variance_summary()
    assert summary["input_dim"] == len(FEATURE_COLUMNS)
    assert 1 <= summary["output_dim"] < len(FEATURE_COLUMNS)
    assert summary["total_explained_variance"] >= 0.949
    assert list(reduced.columns) == pca.output_columns


def test_cnn1d_and_lstm_classifiers_fit_save_load_and_time_bins(
    mock_dataset_dir: Path, tmp_path: Path
) -> None:
    loader = XJTUSYDataLoader(data_dir=mock_dataset_dir, sampling_rate=25600)
    train_units, test_units = loader.split_bearings_by_unit(test_ratio=0.30, seed=42)
    train_df = build_feature_dataset(loader, bearings=train_units)
    test_df = build_feature_dataset(loader, bearings=test_units)

    tr = train_df[(train_df["life_ratio"] >= 0.30) & (train_df["life_ratio"] <= 0.90)]
    te = test_df[(test_df["life_ratio"] >= 0.30) & (test_df["life_ratio"] <= 0.90)]

    # 训练并验证 1D-CNN 与 LSTM 故障分类器
    cnn = CNN1DFaultClassifier(epochs=12, batch_size=16, random_state=42, device="cpu").fit(
        tr, tr["fault_type"]
    )
    lstm = LSTMFaultClassifier(epochs=12, batch_size=16, random_state=42, device="cpu").fit(
        tr, tr["fault_type"]
    )

    for key, clf, filename in (
        ("cnn1d", cnn, "cnn1d_fault_classifier.pt"),
        ("lstm", lstm, "lstm_fault_classifier.pt"),
    ):
        ev = clf.evaluate(te, te["fault_type"])
        assert 0.0 <= ev["accuracy"] <= 1.0
        assert 0.0 <= ev["macro_f1"] <= 1.0
        assert ev["train_time_seconds"] > 0
        assert ev["infer_time_ms_per_sample"] >= 0

        saved_path = clf.save(tmp_path / filename)
        assert saved_path.exists()

        reloaded = load_fault_classifier(key, tmp_path, device="cpu")
        sample_feats = extract_sample_features(loader.load_file("Bearing1_1", 10))
        one = reloaded.classify_one(sample_feats)
        assert one.fault_type in reloaded.classes_
        assert 0.0 <= one.confidence <= 1.0
        assert abs(sum(one.probabilities.values()) - 1.0) < 0.02

        # 验证按生命周期时间分箱评估
        bins = evaluate_predictions_over_time_bins(
            test_df[test_df["life_ratio"] >= 0.20],
            test_df[test_df["life_ratio"] >= 0.20]["fault_type"],
            reloaded.predict(test_df[test_df["life_ratio"] >= 0.20]),
        )
        assert len(bins) >= 3
        assert all(0.0 <= b["accuracy"] <= 1.0 for b in bins)


def test_rul_normalization_comparison_and_model_switching(
    mock_dataset_dir: Path, tmp_path: Path
) -> None:
    loader = XJTUSYDataLoader(data_dir=mock_dataset_dir, sampling_rate=25600)
    train_units, test_units = loader.split_bearings_by_unit(test_ratio=0.30, seed=42)
    train_df = build_feature_dataset(loader, bearings=train_units)
    test_df = build_feature_dataset(loader, bearings=test_units)

    # 1. Max-based vs 工况分组归一化 vs 绝对分钟 RUL 对比
    rul_cmp = compare_rul_normalization_strategies(
        train_df=train_df, test_df=test_df, n_estimators=60, random_state=42
    )
    assert set(rul_cmp.keys()) == {"max_based", "group_minmax", "absolute"}
    for key in ("max_based", "group_minmax", "absolute"):
        assert "rmse" in rul_cmp[key]["relative_scale"]
        assert "mae" in rul_cmp[key]["relative_scale"]
        assert "r2" in rul_cmp[key]["relative_scale"]
    # Max-based 相对归一化在统一尺度上应具备良好的回归性能
    assert rul_cmp["max_based"]["relative_scale"]["mae"] <= 0.25

    # 2. 训练含 1D-CNN 与 LSTM 的完整 ModelBundle，并验证智能体动态切换分类模型
    art_dir = tmp_path / "full_models"
    pipeline = TrainingPipeline(
        loader=loader,
        artifact_dir=art_dir,
        random_state=42,
        train_deep_models=True,
        deep_epochs=8,
        deep_batch_size=16,
    )
    pipeline.run()

    reloaded = ModelBundle.load(art_dir)
    assert reloaded.available_classifiers() == ["rf", "cnn1d", "lstm"]

    toolkit = EquipmentToolkit(
        loader=loader, bundle=reloaded, config=AppConfig(openai_api_key="")
    )
    for model_key, expected_name in (
        ("rf", "RandomForestClassifier"),
        ("cnn1d", "CNN1DFaultClassifier"),
        ("lstm", "LSTMFaultClassifier"),
    ):
        toolkit.set_classifier_model(model_key)
        assert toolkit.get_active_classifier_model() == model_key
        res = toolkit.classify_fault("Bearing1_1", 12)
        assert res["model_name"] == expected_name
        assert 0.0 <= res["confidence"] <= 1.0
