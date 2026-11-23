"""模拟西交大 XJTU-SY 滚动轴承振动数据生成脚本。

在没有真实 XJTU-SY 数据集时，按照完全一致的目录结构与 CSV 格式生成模拟加速寿命试验数据：
- 目录结构：`<output>/<工况>/<轴承编号>/<1.csv .. N.csv>`
- CSV 列名：`Horizontal_vibration_signals,Vertical_vibration_signals`
- 默认采样率：25.6 kHz，单文件默认 32,768 点（与真实 XJTU-SY 完全一致）

物理建模思路：
- 每个轴承经历“健康平稳期 → 早期微弱冲击 → 中期故障特征显著 → 晚期剧烈退化”全生命周期；
- 不同故障类型（外圈故障、内圈故障、保持架故障、复合故障）注入不同的特征冲击频率、
  共振载波频带、幅值调制规律以及水平/垂直通道能量比，使无监督异常检测、故障分类与
  RUL 回归均能真实反映工业振动信号特征。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# 默认生成 12 个轴承，覆盖全部 3 个标准工况与 4 种故障类型（每种故障 >= 2 个轴承便于按轴承划分测试集）
MOCK_BEARINGS: list[dict[str, object]] = [
    # 工况 1：35Hz12kN (转频 35 Hz)
    {"condition": "35Hz12kN", "bearing_id": "Bearing1_1", "fault_type": "外圈故障", "fr": 35.0, "life_scale": 1.00},
    {"condition": "35Hz12kN", "bearing_id": "Bearing1_2", "fault_type": "外圈故障", "fr": 35.0, "life_scale": 1.15},
    {"condition": "35Hz12kN", "bearing_id": "Bearing1_4", "fault_type": "保持架故障", "fr": 35.0, "life_scale": 0.95},
    {"condition": "35Hz12kN", "bearing_id": "Bearing1_5", "fault_type": "复合故障", "fr": 35.0, "life_scale": 0.90},
    # 工况 2：37.5Hz11kN (转频 37.5 Hz)
    {"condition": "37.5Hz11kN", "bearing_id": "Bearing2_1", "fault_type": "内圈故障", "fr": 37.5, "life_scale": 1.10},
    {"condition": "37.5Hz11kN", "bearing_id": "Bearing2_2", "fault_type": "外圈故障", "fr": 37.5, "life_scale": 1.05},
    {"condition": "37.5Hz11kN", "bearing_id": "Bearing2_3", "fault_type": "保持架故障", "fr": 37.5, "life_scale": 1.00},
    {"condition": "37.5Hz11kN", "bearing_id": "Bearing2_4", "fault_type": "外圈故障", "fr": 37.5, "life_scale": 0.92},
    # 工况 3：40Hz10kN (转频 40 Hz)
    {"condition": "40Hz10kN", "bearing_id": "Bearing3_1", "fault_type": "外圈故障", "fr": 40.0, "life_scale": 1.20},
    {"condition": "40Hz10kN", "bearing_id": "Bearing3_2", "fault_type": "复合故障", "fr": 40.0, "life_scale": 0.88},
    {"condition": "40Hz10kN", "bearing_id": "Bearing3_3", "fault_type": "内圈故障", "fr": 40.0, "life_scale": 1.08},
    {"condition": "40Hz10kN", "bearing_id": "Bearing3_4", "fault_type": "内圈故障", "fr": 40.0, "life_scale": 0.96},
]


def _synthesize_impulse_train(
    t: np.ndarray,
    fault_freq: float,
    carrier_freq: float,
    damping: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成周期性衰减冲击脉冲序列（模拟滚动体经过点蚀/剥落缺陷激发的高频结构共振）。"""
    period = 1.0 / fault_freq
    # 加入轻微随机滑差相位，更贴近真实轴承物理特性
    phase_offset = rng.uniform(0.0, period)
    local_t = np.mod(t + phase_offset, period)
    envelope = np.exp(-damping * local_t)
    return envelope * np.sin(2.0 * np.pi * carrier_freq * local_t)


def synthesize_bearing_file(
    t: np.ndarray,
    fr: float,
    fault_type: str,
    life_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """合成单个时间点（1 分钟窗口）的水平与垂直振动信号。"""
    n = len(t)
    # 1. 健康基线：转频及其二倍频 + 平稳高斯测量噪声
    base_h = (
        0.18 * np.sin(2.0 * np.pi * fr * t + rng.uniform(0, 2 * np.pi))
        + 0.08 * np.sin(2.0 * np.pi * (2.0 * fr) * t + rng.uniform(0, 2 * np.pi))
        + rng.normal(0.0, 0.22, size=n)
    )
    base_v = (
        0.16 * np.sin(2.0 * np.pi * fr * t + rng.uniform(0, 2 * np.pi))
        + 0.07 * np.sin(2.0 * np.pi * (2.0 * fr) * t + rng.uniform(0, 2 * np.pi))
        + rng.normal(0.0, 0.20, size=n)
    )

    # 2. 退化强度随寿命非线性递增（前 25% 接近全新健康，之后逐步显现并加速恶化）
    if life_ratio <= 0.22:
        deg = 0.02 * (life_ratio / 0.22)
    else:
        progress = (life_ratio - 0.22) / 0.78
        deg = 0.02 + 2.6 * (progress**1.85)

    # LDK UER204 滚动轴承近似故障特征频率系数（XJTU-SY 试验轴承型号）
    bpfo = 3.08 * fr  # 外圈故障特征频率
    bpfi = 4.92 * fr  # 内圈故障特征频率
    ftf = 0.38 * fr   # 保持架故障特征频率

    if fault_type == "外圈故障":
        # 外圈固定在承载区：稳定周期冲击，中高频共振（~3800Hz），水平方向能量明显大于垂直方向
        imp = _synthesize_impulse_train(t, bpfo, carrier_freq=3800.0, damping=950.0, rng=rng)
        fault_h = deg * (1.25 * imp + 0.25 * np.sin(2.0 * np.pi * bpfo * t))
        fault_v = deg * (0.55 * imp + 0.12 * np.sin(2.0 * np.pi * bpfo * t))
        extra_noise_h = rng.normal(0.0, 0.10 * deg, size=n)
        extra_noise_v = rng.normal(0.0, 0.06 * deg, size=n)

    elif fault_type == "内圈故障":
        # 内圈随轴旋转：受转频强烈幅值调制，高频共振（~5200Hz），垂直方向响应更显著
        mod = 1.0 + 0.75 * np.cos(2.0 * np.pi * fr * t)
        imp = _synthesize_impulse_train(t, bpfi, carrier_freq=5200.0, damping=1400.0, rng=rng)
        fault_h = deg * (0.60 * mod * imp)
        fault_v = deg * (1.35 * mod * imp + 0.22 * np.sin(2.0 * np.pi * bpfi * t))
        extra_noise_h = rng.normal(0.0, 0.08 * deg, size=n)
        extra_noise_v = rng.normal(0.0, 0.14 * deg, size=n)

    elif fault_type == "保持架故障":
        # 保持架磨损：低频 FTF 调制 + 中低频共振（~1750Hz）+ 宽带摩擦随机冲击
        imp = _synthesize_impulse_train(t, ftf * 3.0, carrier_freq=1750.0, damping=420.0, rng=rng)
        low_osc = np.sin(2.0 * np.pi * ftf * t) + 0.6 * np.sin(2.0 * np.pi * (2.0 * ftf) * t)
        fault_h = deg * (0.85 * imp + 0.55 * low_osc)
        fault_v = deg * (0.80 * imp + 0.50 * low_osc)
        extra_noise_h = rng.normal(0.0, 0.28 * deg, size=n)
        extra_noise_v = rng.normal(0.0, 0.28 * deg, size=n)

    else:  # 复合故障
        # 多部位耦合退化：外圈 + 内圈 + 极高频 8.2kHz 摩擦共振，全频带剧烈抬升
        imp_o = _synthesize_impulse_train(t, bpfo, carrier_freq=3600.0, damping=800.0, rng=rng)
        imp_i = _synthesize_impulse_train(t, bpfi, carrier_freq=8200.0, damping=1600.0, rng=rng)
        mod = 1.0 + 0.5 * np.cos(2.0 * np.pi * fr * t)
        fault_h = deg * (0.95 * imp_o + 0.90 * mod * imp_i)
        fault_v = deg * (0.95 * imp_o + 0.95 * mod * imp_i)
        extra_noise_h = rng.normal(0.0, 0.35 * deg, size=n)
        extra_noise_v = rng.normal(0.0, 0.35 * deg, size=n)

    h_sig = base_h + fault_h + extra_noise_h
    v_sig = base_v + fault_v + extra_noise_v
    return h_sig, v_sig


def generate_mock_xjtu_dataset(
    output_dir: str | Path,
    files_per_bearing: int = 24,
    samples_per_file: int = 32768,
    sampling_rate: int = 25600,
    seed: int = 42,
    bearings_spec: list[dict[str, object]] | None = None,
) -> Path:
    """生成完整的模拟 XJTU-SY 目录树与 CSV 文件。"""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(samples_per_file, dtype=np.float64) / float(sampling_rate)
    specs = bearings_spec or MOCK_BEARINGS

    total_files = 0
    for spec in specs:
        condition = str(spec["condition"])
        bearing_id = str(spec["bearing_id"])
        fault_type = str(spec["fault_type"])
        fr = float(spec["fr"])
        life_scale = float(spec.get("life_scale", 1.0))

        n_files = max(10, int(round(files_per_bearing * life_scale)))
        bearing_dir = out_root / condition / bearing_id
        bearing_dir.mkdir(parents=True, exist_ok=True)

        # 清理旧文件，避免重复生成残留多余编号
        for old_csv in bearing_dir.glob("*.csv"):
            old_csv.unlink()

        for idx in range(1, n_files + 1):
            life_ratio = float(idx) / float(n_files)
            h_sig, v_sig = synthesize_bearing_file(
                t=t,
                fr=fr,
                fault_type=fault_type,
                life_ratio=life_ratio,
                rng=rng,
            )
            data = np.column_stack((h_sig, v_sig))
            csv_path = bearing_dir / f"{idx}.csv"
            np.savetxt(
                csv_path,
                data,
                delimiter=",",
                header="Horizontal_vibration_signals,Vertical_vibration_signals",
                comments="",
                fmt="%.5f",
            )
            total_files += 1

    print(
        f"[模拟数据生成完成] 目录：{out_root.resolve()} | "
        f"轴承数：{len(specs)} | 总 CSV 文件数：{total_files} | "
        f"单文件采样点数：{samples_per_file}（{sampling_rate} Hz）"
    )
    return out_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成符合 XJTU-SY 标准目录结构与双通道格式的模拟轴承振动数据"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/xjtu_sy",
        help="输出根目录（默认：data/xjtu_sy）",
    )
    parser.add_argument(
        "--files-per-bearing",
        type=int,
        default=24,
        help="每个轴承的基准采样文件数/总寿命分钟数（默认：24）",
    )
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=32768,
        help="单个 CSV 文件的采样点数（默认：32768，与 XJTU-SY 一致）",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=25600,
        help="采样频率 Hz（默认：25600）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认：42）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_mock_xjtu_dataset(
        output_dir=args.output,
        files_per_bearing=args.files_per_bearing,
        samples_per_file=args.samples_per_file,
        sampling_rate=args.sampling_rate,
        seed=args.seed,
    )
