"""轴承时频特征相关性分析与可选 PCA 降维工具模块。

在保留原有 54 维双通道时频特征的前提下，提供：
1. 特征间皮尔逊相关性矩阵与高相关冗余特征对识别；
2. 各特征与剩余寿命（`rul_normalized`）的相关性排序（Top-K 退化敏感特征）；
3. 基于 `StandardScaler + PCA` 的可选降维工具，支持按方差解释率阈值（如 0.95）
   或固定主成分维数进行无监督降维与方差贡献统计。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from equipment_agent.features.extractor import FEATURE_COLUMNS


@dataclass
class FeatureCorrelationReport:
    """特征相关性分析结果。"""

    feature_columns: list[str]
    corr_matrix: pd.DataFrame
    high_corr_pairs: list[dict[str, Any]]
    rul_correlations: list[dict[str, Any]]

    def to_summary_dict(self, top_k_pairs: int = 15, top_k_rul: int = 12) -> dict[str, Any]:
        return {
            "total_features": len(self.feature_columns),
            "high_corr_pair_count": len(self.high_corr_pairs),
            "top_high_corr_pairs": self.high_corr_pairs[:top_k_pairs],
            "top_rul_correlated_features": self.rul_correlations[:top_k_rul],
        }


def compute_feature_correlation(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    high_corr_threshold: float = 0.90,
    rul_column: str = "rul_normalized",
) -> FeatureCorrelationReport:
    """计算 54 维时频特征的相关系数矩阵、高相关冗余对以及与 RUL 的相关性排名。"""
    cols = [c for c in (feature_columns or FEATURE_COLUMNS) if c in df.columns]
    if not cols:
        raise ValueError("输入 DataFrame 中未找到有效特征列。")

    feat_df = df[cols].astype(np.float64)
    corr_mat = feat_df.corr(method="pearson").fillna(0.0)

    # 提取上三角高相关特征对
    high_pairs: list[dict[str, Any]] = []
    n = len(cols)
    values = corr_mat.to_numpy()
    for i in range(n):
        for j in range(i + 1, n):
            val = float(values[i, j])
            if abs(val) >= high_corr_threshold:
                high_pairs.append(
                    {
                        "feature_a": cols[i],
                        "feature_b": cols[j],
                        "corr": round(val, 4),
                        "abs_corr": round(abs(val), 4),
                    }
                )
    high_pairs.sort(key=lambda item: item["abs_corr"], reverse=True)

    # 计算与 RUL 的皮尔逊相关系数（若存在目标列）
    rul_corrs: list[dict[str, Any]] = []
    if rul_column in df.columns:
        target = pd.to_numeric(df[rul_column], errors="coerce").to_numpy(dtype=np.float64)
        t_std = float(np.std(target))
        if t_std > 1e-12:
            for col in cols:
                vec = feat_df[col].to_numpy(dtype=np.float64)
                if float(np.std(vec)) <= 1e-12:
                    c_val = 0.0
                else:
                    c_val = float(np.corrcoef(vec, target)[0, 1])
                    if np.isnan(c_val):
                        c_val = 0.0
                rul_corrs.append(
                    {
                        "feature": col,
                        "corr_with_rul": round(c_val, 4),
                        "abs_corr": round(abs(c_val), 4),
                    }
                )
            rul_corrs.sort(key=lambda item: item["abs_corr"], reverse=True)

    return FeatureCorrelationReport(
        feature_columns=cols,
        corr_matrix=corr_mat,
        high_corr_pairs=high_pairs,
        rul_correlations=rul_corrs,
    )


class PCADimensionalityReducer:
    """可选的时频特征 PCA 降维器（内置 Z-Score 标准化）。"""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        n_components: int | float = 0.95,
        random_state: int = 42,
    ) -> None:
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self._fitted = False

    def _to_matrix(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            return x[self.feature_columns].to_numpy(dtype=np.float64)
        arr = np.asarray(x, dtype=np.float64)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    @property
    def output_columns(self) -> list[str]:
        if not self._fitted:
            return []
        return [f"pca_{i + 1}" for i in range(self.pca.n_components_)]

    def fit(self, x: pd.DataFrame | np.ndarray) -> "PCADimensionalityReducer":
        mat = self._to_matrix(x)
        scaled = self.scaler.fit_transform(mat)
        self.pca.fit(scaled)
        self._fitted = True
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("PCADimensionalityReducer 尚未拟合，请先调用 fit()。")
        mat = self._to_matrix(x)
        scaled = self.scaler.transform(mat)
        reduced = self.pca.transform(scaled)
        return pd.DataFrame(reduced, columns=self.output_columns)

    def fit_transform(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        self.fit(x)
        return self.transform(x)

    def explained_variance_summary(self) -> dict[str, Any]:
        """返回降维后的主成分数、各主成分方差解释率及累计解释率。"""
        if not self._fitted:
            raise RuntimeError("PCADimensionalityReducer 尚未拟合。")
        ratios = [round(float(v), 4) for v in self.pca.explained_variance_ratio_]
        cum_ratios = [round(float(v), 4) for v in np.cumsum(self.pca.explained_variance_ratio_)]
        return {
            "input_dim": len(self.feature_columns),
            "output_dim": int(self.pca.n_components_),
            "target_n_components": self.n_components,
            "explained_variance_ratio": ratios,
            "cumulative_variance_ratio": cum_ratios,
            "total_explained_variance": round(float(np.sum(self.pca.explained_variance_ratio_)), 4),
        }
