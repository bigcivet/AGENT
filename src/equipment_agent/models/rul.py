"""基于随机森林回归（Random Forest Regressor）的剩余寿命（RUL）预测基线模型。

参考《轴承剩余寿命预测归一化》：
默认使用 Max-based 归一化相对剩余寿命（范围 [0, 1]，1 表示全新，0 表示完全失效），
可消除不同工况与轴承总寿命尺度差异；同时结合参考平均总寿命换算为估算剩余分钟数，
方便现场运维人员直观理解。

实现 `BaseRULPredictor` 统一接口，后续可无缝替换为 LSTM / GRU / Transformer 时序模型。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from equipment_agent.features.extractor import FEATURE_COLUMNS
from equipment_agent.models.base import BaseRULPredictor


@dataclass
class RULPredictionResult:
    """单时间点剩余寿命预测结果。"""

    rul_ratio: float                # 归一化剩余寿命比例 [0, 1]
    estimated_rul_minutes: float    # 估算剩余运行时间（分钟）
    reference_total_life: float     # 换算所用的参考总寿命（分钟）
    urgency_level: str              # 充裕 / 关注 / 预警 / 紧急
    model_name: str = "RandomForestRegressor"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RandomForestRULPredictor(BaseRULPredictor):
    """随机森林剩余寿命（RUL）预测器（基线模型）。"""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        mode: str = "relative",  # "relative" (归一化 [0,1]) 或 "absolute" (分钟)
        n_estimators: int = 200,
        max_depth: int | None = 16,
        random_state: int = 42,
    ) -> None:
        if mode not in {"relative", "absolute"}:
            raise ValueError("mode 必须为 'relative' 或 'absolute'")
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "reg",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        # 训练集中各工况及全局的平均总寿命（分钟），用于将相对 RUL 换算为参考分钟数
        self.condition_avg_life: dict[str, float] = {}
        self.global_avg_life: float = 120.0
        self._fitted: bool = False

    def _prepare_matrix(self, x: pd.DataFrame | np.ndarray | dict[str, float]) -> np.ndarray:
        if isinstance(x, dict):
            return np.array([[float(x[c]) for c in self.feature_columns]], dtype=np.float64)
        if isinstance(x, pd.DataFrame):
            return x[self.feature_columns].to_numpy(dtype=np.float64)
        arr = np.asarray(x, dtype=np.float64)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        condition_avg_life: dict[str, float] | None = None,
        global_avg_life: float | None = None,
    ) -> "RandomForestRULPredictor":
        mat = self._prepare_matrix(x)
        targets = np.asarray(y, dtype=np.float64)
        self.pipeline.fit(mat, targets)
        if condition_avg_life:
            self.condition_avg_life = {k: float(v) for k, v in condition_avg_life.items()}
        if global_avg_life is not None and global_avg_life > 0:
            self.global_avg_life = float(global_avg_life)
        self._fitted = True
        return self

    def predict(self, x: pd.DataFrame | np.ndarray | dict[str, float]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RandomForestRULPredictor 尚未训练，请先调用 fit() 或 load()。")
        mat = self._prepare_matrix(x)
        preds = self.pipeline.predict(mat)
        if self.mode == "relative":
            return np.clip(preds, 0.0, 1.0)
        return np.maximum(preds, 0.0)

    @staticmethod
    def _urgency_from_ratio(rul_ratio: float) -> str:
        if rul_ratio >= 0.55:
            return "充裕"
        if rul_ratio >= 0.30:
            return "关注"
        if rul_ratio >= 0.12:
            return "预警"
        return "紧急"

    def predict_one(
        self,
        features: dict[str, float] | pd.Series | np.ndarray,
        condition: str | None = None,
        reference_total_life: float | None = None,
    ) -> RULPredictionResult:
        """对单个采样文件预测剩余寿命（同时给出归一化比例与参考剩余分钟数）。"""
        if isinstance(features, pd.Series):
            features = features.to_dict()

        ref_life = (
            float(reference_total_life)
            if reference_total_life and reference_total_life > 0
            else self.condition_avg_life.get(condition or "", self.global_avg_life)
        )
        raw_val = float(self.predict(features)[0])

        if self.mode == "relative":
            rul_ratio = float(np.clip(raw_val, 0.0, 1.0))
            est_minutes = rul_ratio * ref_life
        else:
            est_minutes = max(raw_val, 0.0)
            rul_ratio = float(np.clip(est_minutes / max(ref_life, 1.0), 0.0, 1.0))

        return RULPredictionResult(
            rul_ratio=round(rul_ratio, 4),
            estimated_rul_minutes=round(est_minutes, 1),
            reference_total_life=round(ref_life, 1),
            urgency_level=self._urgency_from_ratio(rul_ratio),
        )

    def evaluate(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> dict[str, float]:
        y_true = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(x)
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
        return {
            "mode": self.mode,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_columns": self.feature_columns,
                "mode": self.mode,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "pipeline": self.pipeline,
                "condition_avg_life": self.condition_avg_life,
                "global_avg_life": self.global_avg_life,
                "fitted": self._fitted,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestRULPredictor":
        payload = joblib.load(path)
        obj = cls(
            feature_columns=payload["feature_columns"],
            mode=payload["mode"],
            n_estimators=payload["n_estimators"],
            max_depth=payload["max_depth"],
            random_state=payload["random_state"],
        )
        obj.pipeline = payload["pipeline"]
        obj.condition_avg_life = payload.get("condition_avg_life", {})
        obj.global_avg_life = payload.get("global_avg_life", 120.0)
        obj._fitted = payload["fitted"]
        return obj
