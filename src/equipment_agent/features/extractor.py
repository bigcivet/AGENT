"""滚动轴承振动信号时域与频域特征提取模块。

参考《轴承数据特征提取参考思路》：
将每个 1 分钟采样文件视为一个时间点，对水平（H）和垂直（V）两个通道分别提取：
1. 时域特征：均值、标准差、均方根(RMS)、峰峰值、绝对均值、偏度、峭度、
   波形因子、峰值因子、脉冲因子、裕度因子。
2. 频域特征（基于实信号 FFT）：主频、主频幅值、频谱重心、均方频率、频率标准差、
   频谱熵，以及 4 个典型工业频带（0-1kHz, 1-3kHz, 3-6kHz, 6-12.8kHz）的能量与能量占比。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from equipment_agent.data.loader import BearingInfo, VibrationSample, XJTUSYDataLoader


_EPS = 1e-12

# 频带划分（Hz），适配 25.6 kHz 采样率（奈奎斯特频率 12.8 kHz）
FREQ_BANDS: list[tuple[str, float, float]] = [
    ("band_0_1k", 0.0, 1000.0),      # 转频、保持架/外圈/内圈基频及低阶谐波
    ("band_1_3k", 1000.0, 3000.0),   # 中低频故障谐波区
    ("band_3_6k", 3000.0, 6000.0),   # 轴承结构高频共振区
    ("band_6_12k", 6000.0, 12800.0), # 早期微弱冲击与高频摩擦噪声区
]

_SINGLE_CHANNEL_FEATURE_NAMES: list[str] = [
    # 时域基本统计与冲击形状特征
    "mean",
    "std",
    "rms",
    "peak_to_peak",
    "abs_mean",
    "skewness",
    "kurtosis",
    "waveform_factor",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    # 频域统计与频带能量特征
    "dominant_freq",
    "dominant_amp",
    "spectral_centroid",
    "msf",
    "freq_std",
    "spectral_entropy",
    "total_spectral_energy",
    "band_0_1k_energy",
    "band_0_1k_ratio",
    "band_1_3k_energy",
    "band_1_3k_ratio",
    "band_3_6k_energy",
    "band_3_6k_ratio",
    "band_6_12k_energy",
    "band_6_12k_ratio",
]

# 双通道完整特征名列表（保证训练与推理特征列顺序严格一致）
FEATURE_COLUMNS: list[str] = (
    [f"h_{name}" for name in _SINGLE_CHANNEL_FEATURE_NAMES]
    + [f"v_{name}" for name in _SINGLE_CHANNEL_FEATURE_NAMES]
    + ["hv_rms_ratio", "hv_corr"]
)


def compute_fft_spectrum(
    signal: np.ndarray, sampling_rate: int = 25600
) -> tuple[np.ndarray, np.ndarray]:
    """计算去直流分量后的单边幅值谱 `(freqs, amplitudes)`。"""
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.array([0.0]), np.array([0.0])

    x_centered = x - np.mean(x)
    fft_vals = np.fft.rfft(x_centered)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sampling_rate))
    amplitudes = (2.0 / n) * np.abs(fft_vals)
    amplitudes[0] /= 2.0
    return freqs, amplitudes


def extract_channel_features(
    signal: np.ndarray, sampling_rate: int = 25600, prefix: str = ""
) -> dict[str, float]:
    """提取单通道振动信号的时域 + 频域特征字典。"""
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        raise ValueError("输入振动信号为空，无法提取特征。")

    # ---- 1. 时域特征 ----
    mean_val = float(np.mean(x))
    std_val = float(np.std(x))
    rms_val = float(np.sqrt(np.mean(x**2)))
    peak_val = float(np.max(np.abs(x)))
    p2p_val = float(np.max(x) - np.min(x))
    abs_mean = float(np.mean(np.abs(x)))
    sqrt_amp = float(np.mean(np.sqrt(np.abs(x))) ** 2)

    # 常数信号时 skew/kurtosis 会返回 NaN，做安全兜底
    skew_val = float(skew(x, bias=False)) if std_val > _EPS else 0.0
    kurt_val = float(kurtosis(x, fisher=False, bias=False)) if std_val > _EPS else 3.0

    waveform_factor = rms_val / (abs_mean + _EPS)
    crest_factor = peak_val / (rms_val + _EPS)
    impulse_factor = peak_val / (abs_mean + _EPS)
    clearance_factor = peak_val / (sqrt_amp + _EPS)

    # ---- 2. 频域特征 ----
    freqs, amps = compute_fft_spectrum(x, sampling_rate=sampling_rate)
    power = amps**2
    total_energy = float(np.sum(power))

    # 排除 0Hz 直流点寻找主频
    if len(amps) > 1:
        dom_idx = int(np.argmax(amps[1:])) + 1
    else:
        dom_idx = 0
    dominant_freq = float(freqs[dom_idx])
    dominant_amp = float(amps[dom_idx])

    amp_sum = float(np.sum(amps)) + _EPS
    spectral_centroid = float(np.sum(freqs * amps) / amp_sum)
    msf = float(np.sum((freqs**2) * power) / (total_energy + _EPS))
    freq_var = float(np.sum(((freqs - spectral_centroid) ** 2) * amps) / amp_sum)
    freq_std = float(np.sqrt(max(freq_var, 0.0)))

    # 归一化频谱熵（衡量频谱能量分散/集中程度）
    p_dist = power / (total_energy + _EPS)
    p_nonzero = p_dist[p_dist > _EPS]
    spectral_entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero))) if p_nonzero.size else 0.0

    raw: dict[str, float] = {
        "mean": mean_val,
        "std": std_val,
        "rms": rms_val,
        "peak_to_peak": p2p_val,
        "abs_mean": abs_mean,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "waveform_factor": waveform_factor,
        "crest_factor": crest_factor,
        "impulse_factor": impulse_factor,
        "clearance_factor": clearance_factor,
        "dominant_freq": dominant_freq,
        "dominant_amp": dominant_amp,
        "spectral_centroid": spectral_centroid,
        "msf": msf,
        "freq_std": freq_std,
        "spectral_entropy": spectral_entropy,
        "total_spectral_energy": total_energy,
    }

    # 分频带能量与占比
    for band_name, f_low, f_high in FREQ_BANDS:
        mask = (freqs >= f_low) & (freqs < f_high)
        band_energy = float(np.sum(power[mask]))
        raw[f"{band_name}_energy"] = band_energy
        raw[f"{band_name}_ratio"] = band_energy / (total_energy + _EPS)

    if prefix:
        return {f"{prefix}{k}": v for k, v in raw.items()}
    return raw


def extract_dual_channel_features(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    sampling_rate: int = 25600,
) -> dict[str, float]:
    """提取水平与垂直双通道联合特征，包含通道间能量比与相关系数。"""
    h_feats = extract_channel_features(horizontal, sampling_rate=sampling_rate, prefix="h_")
    v_feats = extract_channel_features(vertical, sampling_rate=sampling_rate, prefix="v_")

    feats = {**h_feats, **v_feats}
    feats["hv_rms_ratio"] = h_feats["h_rms"] / (v_feats["v_rms"] + _EPS)

    h_std = float(np.std(horizontal))
    v_std = float(np.std(vertical))
    if h_std > _EPS and v_std > _EPS:
        min_len = min(len(horizontal), len(vertical))
        corr = float(np.corrcoef(horizontal[:min_len], vertical[:min_len])[0, 1])
    else:
        corr = 0.0
    feats["hv_corr"] = 0.0 if np.isnan(corr) else corr
    return feats


def extract_sample_features(sample: VibrationSample) -> dict[str, float | str | int]:
    """提取单个 `VibrationSample` 的特征并附带元数据与 RUL 标签。"""
    feats = extract_dual_channel_features(
        sample.horizontal, sample.vertical, sampling_rate=sample.sampling_rate
    )
    record: dict[str, float | str | int] = {
        "condition": sample.condition,
        "bearing_id": sample.bearing_id,
        "file_index": sample.file_index,
        "total_files": sample.total_files,
        "life_ratio": sample.life_ratio,
        "fault_type": sample.fault_type,
        "rul_minutes": sample.rul_minutes,
        "rul_normalized": sample.rul_normalized,
    }
    record.update(feats)
    return record


def build_feature_dataset(
    loader: XJTUSYDataLoader,
    bearings: Iterable[BearingInfo] | None = None,
    step: int = 1,
    life_ratio_range: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """批量提取多个轴承的时频特征表，用于训练或回测。

    Args:
        loader: XJTU-SY 数据加载器。
        bearings: 指定要提取的轴承列表；为 None 时提取全部轴承。
        step: 采样步长（1 表示每个文件都提取）。
        life_ratio_range: 仅保留处于 `[min_ratio, max_ratio]` 寿命区间的文件。
    """
    target_bearings = list(bearings) if bearings is not None else loader.list_bearings()
    min_ratio, max_ratio = life_ratio_range
    rows: list[dict[str, float | str | int]] = []

    for info in target_bearings:
        for sample in loader.iter_bearing_stream(info.bearing_id, step=step):
            if min_ratio <= sample.life_ratio <= max_ratio:
                rows.append(extract_sample_features(sample))

    return pd.DataFrame(rows)
