"""数据加载器与时频特征提取模块单元测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from equipment_agent.data.loader import XJTUSYDataLoader, read_vibration_csv
from equipment_agent.features.extractor import (
    FEATURE_COLUMNS,
    extract_dual_channel_features,
    extract_sample_features,
)


def test_loader_scans_xjtu_structure(mock_dataset_dir: Path) -> None:
    loader = XJTUSYDataLoader(data_dir=mock_dataset_dir, sampling_rate=25600)
    assert loader.is_available
    bearings = loader.list_bearings()
    assert len(bearings) == 12
    assert set(loader.list_conditions()) == {"35Hz12kN", "37.5Hz11kN", "40Hz10kN"}

    b = loader.get_bearing("Bearing1_1")
    assert b.condition == "35Hz12kN"
    assert b.fault_type == "外圈故障"
    assert b.total_life_minutes >= 10

    sample = loader.load_file("Bearing1_1", 1)
    assert len(sample.horizontal) == 2048
    assert len(sample.vertical) == 2048
    assert sample.rul_minutes == b.total_life_minutes - 1
    assert 0.0 <= sample.rul_normalized <= 1.0


def test_read_headerless_csv(tmp_path: Path) -> None:
    csv_file = tmp_path / "headerless.csv"
    arr = np.column_stack((np.linspace(0, 1, 64), np.linspace(1, 2, 64)))
    np.savetxt(csv_file, arr, delimiter=",", fmt="%.5f")

    h, v = read_vibration_csv(csv_file)
    assert len(h) == 64
    assert len(v) == 64
    assert np.isclose(h[-1], 1.0, atol=1e-4)
    assert np.isclose(v[0], 1.0, atol=1e-4)


def test_unit_level_split_no_leakage(mock_dataset_dir: Path) -> None:
    loader = XJTUSYDataLoader(data_dir=mock_dataset_dir)
    train_units, test_units = loader.split_bearings_by_unit(test_ratio=0.3, seed=42)
    train_ids = {b.bearing_id for b in train_units}
    test_ids = {b.bearing_id for b in test_units}
    assert len(train_ids) > 0
    assert len(test_ids) > 0
    assert train_ids.isdisjoint(test_ids)


def test_time_and_freq_feature_extraction(mock_dataset_dir: Path) -> None:
    loader = XJTUSYDataLoader(data_dir=mock_dataset_dir)
    early_sample = loader.load_file("Bearing1_1", 1)
    late_sample = loader.load_file("Bearing1_1", loader.get_bearing("Bearing1_1").total_life_minutes)

    early_feats = extract_sample_features(early_sample)
    late_feats = extract_sample_features(late_sample)

    for col in FEATURE_COLUMNS:
        assert col in early_feats
        assert np.isfinite(early_feats[col])
        assert np.isfinite(late_feats[col])

    # 晚期退化样本的 RMS 与峭度应显著高于早期健康样本
    assert late_feats["h_rms"] > early_feats["h_rms"] * 1.5
    assert late_feats["h_kurtosis"] > early_feats["h_kurtosis"]

    # 纯正弦信号主频提取验证
    fs = 25600
    t = np.arange(4096) / fs
    sine_h = np.sin(2.0 * np.pi * 1000.0 * t)
    sine_v = np.cos(2.0 * np.pi * 2500.0 * t)
    feats = extract_dual_channel_features(sine_h, sine_v, sampling_rate=fs)
    assert abs(feats["h_dominant_freq"] - 1000.0) < 15.0
    assert abs(feats["v_dominant_freq"] - 2500.0) < 15.0
