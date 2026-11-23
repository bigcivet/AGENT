"""基于孤立森林（Isolation Forest）的无监督异常检测模块。

典型用法：
1. 仅使用轴承早期健康阶段（如前 20%~30% 生命周期）的时频特征拟合模型；
2. 逐文件实时读取后续振动数据，提取特征后调用 `detect_one()` 计算异常分数与健康指数。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from equipment_agent.features.extractor import FEATURE_COLUMNS
from equipment_agent.models.base import BaseAnomalyDetector


@dataclass
class AnomalyResult:
    """单时间点异常检测结果。"""

    is_anomaly: bool
    anomaly_score: float       # > 0 表示高于决策阈值（越大约异常）
    health_index: float        # 0 ~ 100，100 表示完全健康
    severity_hint: str         # 正常 / 轻度异常 / 中度异常 / 严重异常

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IsolationForestDetector(BaseAnomalyDetector):
    """封装 StandardScaler + IsolationForest 的轴承振动异常检测器。"""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "iforest",
                    IsolationForest(
                        n_estimators=n_estimators,
                        contamination=contamination,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        # 训练期健康样本的异常分数统计，用于把原始分数映射为直观的 0~100 健康指数
        self._baseline_score_mean: float = -0.1
        self._baseline_score_std: float = 0.05
        self._fitted: bool = False

    def _prepare_matrix(self, x: pd.DataFrame | np.ndarray | dict[str, float]) -> np.ndarray:
        if isinstance(x, dict):
            return np.array([[float(x[c]) for c in self.feature_columns]], dtype=np.float64)
        if isinstance(x, pd.DataFrame):
            return x[self.feature_columns].to_numpy(dtype=np.float64)
        arr = np.asarray(x, dtype=np.float64)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    def fit(self, x: pd.DataFrame | np.ndarray) -> "IsolationForestDetector":
        """基于健康阶段特征矩阵拟合孤立森林模型。"""
        mat = self._prepare_matrix(x)
        self.pipeline.fit(mat)
        # sklearn decision_function: >0 偏正常，<0 偏异常；取负值作为 anomaly_score（越大越异常）
        train_scores = -self.pipeline.decision_function(mat)
        self._baseline_score_mean = float(np.mean(train_scores))
        self._baseline_score_std = float(max(np.std(train_scores), 1e-3))
        self._fitted = True
        return self

    def score(self, x: pd.DataFrame | np.ndarray | dict[str, float]) -> np.ndarray:
        """计算异常分数：> 0 代表判定为异常，数值越大表示偏离健康基线越严重。"""
        if not self._fitted:
            raise RuntimeError("IsolationForestDetector 尚未训练，请先调用 fit() 或 load()。")
        mat = self._prepare_matrix(x)
        return -self.pipeline.decision_function(mat)

    def predict(self, x: pd.DataFrame | np.ndarray | dict[str, float]) -> np.ndarray:
        """返回布尔数组，True 表示异常，False 表示正常。"""
        if not self._fitted:
            raise RuntimeError("IsolationForestDetector 尚未训练，请先调用 fit() 或 load()。")
        mat = self._prepare_matrix(x)
        # sklearn IsolationForest.predict: -1 表示异常，1 表示正常
        raw_pred = self.pipeline.predict(mat)
        return raw_pred == -1

    def score_to_health_index(self, anomaly_score: float) -> float:
        """将异常分数平滑映射为 0~100 的设备健康指数。"""
        # 当分数接近基线均值时约 95~100 分；超过 0 后快速衰减
        excess = max(anomaly_score - self._baseline_score_mean, 0.0)
        scale = max(abs(self._baseline_score_mean) + 2.0 * self._baseline_score_std, 0.12)
        health = 100.0 * np.exp(-1.8 * (excess / scale))
        return float(np.clip(health, 0.0, 100.0))

    @staticmethod
    def _severity_from_score(is_anomaly: bool, anomaly_score: float, health_index: float) -> str:
        if not is_anomaly and health_index >= 70.0:
            return "正常"
        if health_index >= 55.0 and anomaly_score < 0.08:
            return "轻度异常"
        if health_index >= 30.0 and anomaly_score < 0.16:
            return "中度异常"
        return "严重异常"

    def detect_one(self, features: dict[str, float] | pd.Series | np.ndarray) -> AnomalyResult:
        """对单个采样文件的特征字典执行异常检测并返回结构化结果。"""
        if isinstance(features, pd.Series):
            features = features.to_dict()
        is_anom = bool(self.predict(features)[0])
        anom_score = float(self.score(features)[0])
        health_idx = self.score_to_health_index(anom_score)
        severity = self._severity_from_score(is_anom, anom_score, health_idx)
        return AnomalyResult(
            is_anomaly=is_anom,
            anomaly_score=round(anom_score, 4),
            health_index=round(health_idx, 2),
            severity_hint=severity,
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_columns": self.feature_columns,
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "pipeline": self.pipeline,
                "baseline_score_mean": self._baseline_score_mean,
                "baseline_score_std": self._baseline_score_std,
                "fitted": self._fitted,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestDetector":
        payload = joblib.load(path)
        obj = cls(
            feature_columns=payload["feature_columns"],
            contamination=payload["contamination"],
            n_estimators=payload["n_estimators"],
            random_state=payload["random_state"],
        )
        obj.pipeline = payload["pipeline"]
        obj._baseline_score_mean = payload["baseline_score_mean"]
        obj._baseline_score_std = payload["baseline_score_std"]
        obj._fitted = payload["fitted"]
        return obj
