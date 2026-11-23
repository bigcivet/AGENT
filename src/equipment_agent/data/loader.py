"""西交大 XJTU-SY 滚动轴承加速寿命试验数据集加载模块。

标准 XJTU-SY 目录结构示例::

    data/xjtu_sy/
    ├── 35Hz12kN/
    │   ├── Bearing1_1/
    │   │   ├── 1.csv
    │   │   ├── 2.csv
    │   │   └── ...
    │   └── Bearing1_2/
    ├── 37.5Hz11kN/
    │   └── Bearing2_1/
    └── 40Hz10kN/
        └── Bearing3_1/

每个 CSV 文件对应 1 分钟的采样窗口，采样频率 25.6 kHz，单文件包含 32,768 个采样点，
含两列振动信号：水平振动 (`Horizontal_vibration_signals`) 与垂直振动 (`Vertical_vibration_signals`)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


# 西交大 XJTU-SY 数据集论文中公布的 15 个轴承工况与失效类型映射
# 参考论文：《XJTU-SY滚动轴承加速寿命试验数据集解读》
XJTU_SY_METADATA: dict[str, dict[str, str | float]] = {
    "Bearing1_1": {"condition": "35Hz12kN", "fault_type": "外圈故障", "rpm": 2100, "load_kn": 12.0},
    "Bearing1_2": {"condition": "35Hz12kN", "fault_type": "外圈故障", "rpm": 2100, "load_kn": 12.0},
    "Bearing1_3": {"condition": "35Hz12kN", "fault_type": "外圈故障", "rpm": 2100, "load_kn": 12.0},
    "Bearing1_4": {"condition": "35Hz12kN", "fault_type": "保持架故障", "rpm": 2100, "load_kn": 12.0},
    "Bearing1_5": {"condition": "35Hz12kN", "fault_type": "复合故障", "rpm": 2100, "load_kn": 12.0},
    "Bearing2_1": {"condition": "37.5Hz11kN", "fault_type": "内圈故障", "rpm": 2250, "load_kn": 11.0},
    "Bearing2_2": {"condition": "37.5Hz11kN", "fault_type": "外圈故障", "rpm": 2250, "load_kn": 11.0},
    "Bearing2_3": {"condition": "37.5Hz11kN", "fault_type": "保持架故障", "rpm": 2250, "load_kn": 11.0},
    "Bearing2_4": {"condition": "37.5Hz11kN", "fault_type": "外圈故障", "rpm": 2250, "load_kn": 11.0},
    "Bearing2_5": {"condition": "37.5Hz11kN", "fault_type": "外圈故障", "rpm": 2250, "load_kn": 11.0},
    "Bearing3_1": {"condition": "40Hz10kN", "fault_type": "外圈故障", "rpm": 2400, "load_kn": 10.0},
    "Bearing3_2": {"condition": "40Hz10kN", "fault_type": "复合故障", "rpm": 2400, "load_kn": 10.0},
    "Bearing3_3": {"condition": "40Hz10kN", "fault_type": "内圈故障", "rpm": 2400, "load_kn": 10.0},
    "Bearing3_4": {"condition": "40Hz10kN", "fault_type": "内圈故障", "rpm": 2400, "load_kn": 10.0},
    "Bearing3_5": {"condition": "40Hz10kN", "fault_type": "外圈故障", "rpm": 2400, "load_kn": 10.0},
}

# 标准列名候选（兼容原始英文列名与常见中文/简写列名）
_HORIZONTAL_ALIASES = {
    "horizontal_vibration_signals",
    "horizontal",
    "horiz",
    "h_signal",
    "水平振动",
    "水平",
}
_VERTICAL_ALIASES = {
    "vertical_vibration_signals",
    "vertical",
    "vert",
    "v_signal",
    "垂直振动",
    "垂直",
}


def _natural_file_index(path: Path) -> int:
    """提取文件名中的数字序号（如 `12.csv` -> 12），保证按采样时间严格递增排序。"""
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


@dataclass
class BearingInfo:
    """单个轴承试验单元的元信息。"""

    condition: str
    bearing_id: str
    folder: Path
    files: list[Path]
    fault_type: str
    sampling_rate: int = 25600

    @property
    def total_life_minutes(self) -> int:
        """轴承总寿命（分钟，XJTU-SY 每分钟采样 1 个文件）。"""
        return len(self.files)

    @property
    def key(self) -> str:
        """唯一标识：`工况/轴承名`。"""
        return f"{self.condition}/{self.bearing_id}"


@dataclass
class VibrationSample:
    """单个 1 分钟采样文件的数据对象。"""

    condition: str
    bearing_id: str
    file_index: int  # 1-based 时间索引（分钟）
    total_files: int
    file_path: Path
    horizontal: np.ndarray
    vertical: np.ndarray
    sampling_rate: int
    fault_type: str

    @property
    def life_ratio(self) -> float:
        """当前处于全寿命周期的比例 [0, 1]。"""
        if self.total_files <= 1:
            return 1.0
        return float(self.file_index) / float(self.total_files)

    @property
    def rul_minutes(self) -> float:
        """真实剩余寿命（分钟）：总文件数 - 当前文件序号。"""
        return float(max(self.total_files - self.file_index, 0))

    @property
    def rul_normalized(self) -> float:
        """Max-based 归一化剩余寿命比例 [0, 1]，1 表示全新，0 表示失效。"""
        if self.total_files <= 0:
            return 0.0
        return float(max(self.total_files - self.file_index, 0)) / float(self.total_files)


def read_vibration_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取单个 XJTU-SY 振动 CSV 文件，返回 `(horizontal, vertical)` 一维浮点数组。

    兼容：
    - 标准 XJTU-SY 表头 `Horizontal_vibration_signals,Vertical_vibration_signals`
    - 常见中英文别名表头
    - 无表头的纯两列数值 CSV
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"振动数据文件不存在：{path}")

    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV 文件列数不足（需要至少 2 列水平/垂直振动信号）：{path}")

    # 判断首行是否本身就是数值（无表头 CSV）
    first_cols_numeric = True
    for col in df.columns[:2]:
        try:
            float(str(col).strip())
        except ValueError:
            first_cols_numeric = False
            break

    if first_cols_numeric:
        df = pd.read_csv(path, header=None)
        h_sig = df.iloc[:, 0].to_numpy(dtype=np.float64)
        v_sig = df.iloc[:, 1].to_numpy(dtype=np.float64)
        return h_sig, v_sig

    # 匹配水平与垂直列名
    col_map = {str(c).strip().lower(): c for c in df.columns}
    h_col = next((col_map[k] for k in _HORIZONTAL_ALIASES if k in col_map), None)
    v_col = next((col_map[k] for k in _VERTICAL_ALIASES if k in col_map), None)

    if h_col is not None and v_col is not None:
        h_sig = pd.to_numeric(df[h_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        v_sig = pd.to_numeric(df[v_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
    else:
        h_sig = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=np.float64)
        v_sig = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().to_numpy(dtype=np.float64)

    min_len = min(len(h_sig), len(v_sig))
    if min_len == 0:
        raise ValueError(f"CSV 文件未包含有效数值信号：{path}")
    return h_sig[:min_len], v_sig[:min_len]


class XJTUSYDataLoader:
    """西交大 XJTU-SY 轴承数据集加载器。

    支持：
    1. 自动扫描标准二级目录（`工况/轴承/1.csv..N.csv`）及平铺轴承目录（`轴承/1.csv..N.csv`）。
    2. 逐文件流式读取（模拟在线实时监测）。
    3. 按单个轴承单元切分训练/测试集，避免同一轴承跨集合导致的数据泄露。
    """

    def __init__(
        self,
        data_dir: str | Path,
        sampling_rate: int = 25600,
        custom_labels: dict[str, str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.custom_labels = custom_labels or {}
        self._bearings: dict[str, BearingInfo] = {}
        self.refresh()

    def refresh(self) -> None:
        """重新扫描数据目录，构建轴承索引。"""
        self._bearings.clear()
        if not self.data_dir.exists():
            return

        # 支持两种目录层级：
        # 1) data_dir/<condition>/<bearing_id>/*.csv（标准 XJTU-SY 结构）
        # 2) data_dir/<bearing_id>/*.csv（简化结构）
        for child in sorted(self.data_dir.iterdir()):
            if not child.is_dir():
                continue

            direct_csvs = sorted(child.glob("*.csv"), key=_natural_file_index)
            if direct_csvs:
                bearing_id = child.name
                meta = XJTU_SY_METADATA.get(bearing_id, {})
                condition = str(meta.get("condition", "default"))
                fault_type = self._resolve_fault_type(bearing_id)
                info = BearingInfo(
                    condition=condition,
                    bearing_id=bearing_id,
                    folder=child,
                    files=direct_csvs,
                    fault_type=fault_type,
                    sampling_rate=self.sampling_rate,
                )
                self._bearings[bearing_id] = info
                continue

            # 标准二级目录：child 为工况名，grandchild 为轴承名
            condition = child.name
            for grandchild in sorted(child.iterdir()):
                if not grandchild.is_dir():
                    continue
                csv_files = sorted(grandchild.glob("*.csv"), key=_natural_file_index)
                if not csv_files:
                    continue
                bearing_id = grandchild.name
                fault_type = self._resolve_fault_type(bearing_id)
                info = BearingInfo(
                    condition=condition,
                    bearing_id=bearing_id,
                    folder=grandchild,
                    files=csv_files,
                    fault_type=fault_type,
                    sampling_rate=self.sampling_rate,
                )
                self._bearings[bearing_id] = info

    def _resolve_fault_type(self, bearing_id: str) -> str:
        if bearing_id in self.custom_labels:
            return self.custom_labels[bearing_id]
        if bearing_id in XJTU_SY_METADATA:
            return str(XJTU_SY_METADATA[bearing_id]["fault_type"])
        return "未知故障"

    @property
    def is_available(self) -> bool:
        """数据目录下是否成功扫描到轴承数据。"""
        return len(self._bearings) > 0

    def list_bearings(self, condition: str | None = None) -> list[BearingInfo]:
        """列出已扫描到的所有轴承信息，可按工况过滤。"""
        items = list(self._bearings.values())
        if condition:
            items = [b for b in items if b.condition == condition]
        return sorted(items, key=lambda b: (b.condition, b.bearing_id))

    def list_conditions(self) -> list[str]:
        """列出所有工况名称。"""
        return sorted({b.condition for b in self._bearings.values()})

    def get_bearing(self, bearing_id: str) -> BearingInfo:
        """按轴承名称（支持 `Bearing1_1` 或 `35Hz12kN/Bearing1_1`）获取元信息。"""
        clean_id = bearing_id.strip().split("/")[-1]
        if clean_id in self._bearings:
            return self._bearings[clean_id]
        # 不区分大小写兜底匹配
        for k, v in self._bearings.items():
            if k.lower() == clean_id.lower():
                return v
        available = ", ".join(sorted(self._bearings.keys())) or "无"
        raise KeyError(f"未找到轴承 `{bearing_id}`。当前可用轴承：{available}")

    def load_file(self, bearing_id: str, file_index: int) -> VibrationSample:
        """读取指定轴承的第 `file_index` 个采样文件（1-based，对应运行第 N 分钟）。"""
        info = self.get_bearing(bearing_id)
        if file_index < 1 or file_index > info.total_life_minutes:
            raise IndexError(
                f"轴承 {info.bearing_id} 的文件索引越界：{file_index}，"
                f"有效范围为 1..{info.total_life_minutes}"
            )
        file_path = info.files[file_index - 1]
        h_sig, v_sig = read_vibration_csv(file_path)
        return VibrationSample(
            condition=info.condition,
            bearing_id=info.bearing_id,
            file_index=file_index,
            total_files=info.total_life_minutes,
            file_path=file_path,
            horizontal=h_sig,
            vertical=v_sig,
            sampling_rate=self.sampling_rate,
            fault_type=info.fault_type,
        )

    def iter_bearing_stream(
        self,
        bearing_id: str,
        start_index: int = 1,
        end_index: int | None = None,
        step: int = 1,
    ) -> Iterator[VibrationSample]:
        """逐文件流式读取指定轴承的振动信号，模拟实时采集监测场景。"""
        info = self.get_bearing(bearing_id)
        stop = min(end_index or info.total_life_minutes, info.total_life_minutes)
        start = max(start_index, 1)
        for idx in range(start, stop + 1, max(step, 1)):
            yield self.load_file(info.bearing_id, idx)

    def split_bearings_by_unit(
        self,
        test_ratio: float = 0.3,
        seed: int = 42,
    ) -> tuple[list[BearingInfo], list[BearingInfo]]:
        """按“单个轴承单元”划分训练集与测试集，避免同一轴承跨集合造成数据泄露。

        策略：优先在每个故障类别内部按比例分层抽取，确保训练集与测试集尽量覆盖各类故障。
        """
        bearings = self.list_bearings()
        if len(bearings) <= 1:
            return bearings, []

        rng = np.random.default_rng(seed)
        by_fault: dict[str, list[BearingInfo]] = {}
        for b in bearings:
            by_fault.setdefault(b.fault_type, []).append(b)

        train_units: list[BearingInfo] = []
        test_units: list[BearingInfo] = []

        for fault_type, group in sorted(by_fault.items()):
            perm = list(rng.permutation(len(group)))
            shuffled = [group[i] for i in perm]
            if len(shuffled) == 1:
                train_units.extend(shuffled)
                continue
            n_test = max(1, int(round(len(shuffled) * test_ratio)))
            n_test = min(n_test, len(shuffled) - 1)
            test_units.extend(shuffled[:n_test])
            train_units.extend(shuffled[n_test:])

        return train_units, test_units
