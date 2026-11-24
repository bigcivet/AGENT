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


def compute_group_normalized_rul(
    df: pd.DataFrame,
    group_max_life: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """按工况分组最大寿命进行 RUL 归一化：`rul_minutes / max_life(condition)`。

    返回 `(normalized_values, group_max_life_dict)`。
    """
    if group_max_life is None:
        group_max_life = (
            df.groupby("condition")["total_files"].max().astype(float).to_dict()
            if not df.empty
            else {}
        )
    fallback_max = max(group_max_life.values()) if group_max_life else 1.0
    denoms = np.array(
        [max(float(group_max_life.get(str(c), fallback_max)), 1.0) for c in df["condition"]],
        dtype=np.float64,
    )
    vals = np.clip(df["rul_minutes"].to_numpy(dtype=np.float64) / denoms, 0.0, 1.0)
    return vals, {str(k): float(v) for k, v in group_max_life.items()}


def compare_rul_normalization_strategies(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    """统一对比三种 RUL 目标归一化策略（对齐语雀《轴承剩余寿命预测归一化》）：

    1. `max_based`：单轴承 Max-based 相对剩余寿命归一化 [0, 1]（推荐）；
    2. `group_minmax`：按工况组内最大寿命归一化；
    3. `absolute`：绝对剩余寿命（分钟），同时折算回相对 [0, 1] 尺度进行公平跨策略对比。
    """
    cols = list(feature_columns or FEATURE_COLUMNS)
    cond_avg_life = train_df.groupby("condition")["total_files"].mean().to_dict()
    global_avg_life = float(train_df["total_files"].mean())

    y_test_rel = test_df["rul_normalized"].to_numpy(dtype=np.float64)
    y_test_abs = test_df["rul_minutes"].to_numpy(dtype=np.float64)
    test_totals = np.maximum(test_df["total_files"].to_numpy(dtype=np.float64), 1.0)

    # 策略 1：Max-based 相对剩余寿命归一化 [0, 1]
    max_model = RandomForestRULPredictor(
        feature_columns=cols,
        mode="relative",
        n_estimators=n_estimators,
        random_state=random_state,
    ).fit(
        train_df,
        train_df["rul_normalized"],
        condition_avg_life=cond_avg_life,
        global_avg_life=global_avg_life,
    )
    pred_max_rel = max_model.predict(test_df)
    pred_max_abs = pred_max_rel * test_totals

    # 策略 2：按工况分组最大寿命归一化
    y_train_grp, group_max = compute_group_normalized_rul(train_df)
    y_test_grp, _ = compute_group_normalized_rul(test_df, group_max_life=group_max)
    grp_model = RandomForestRULPredictor(
        feature_columns=cols,
        mode="relative",
        n_estimators=n_estimators,
        random_state=random_state,
    ).fit(
        train_df,
        y_train_grp,
        condition_avg_life=cond_avg_life,
        global_avg_life=global_avg_life,
    )
    pred_grp_raw = grp_model.predict(test_df)
    # 按对应工况最大寿命还原为分钟，再除以真实总寿命折算到统一 [0,1] 相对尺度对比
    fallback_max = max(group_max.values()) if group_max else global_avg_life
    test_group_denoms = np.array(
        [max(float(group_max.get(str(c), fallback_max)), 1.0) for c in test_df["condition"]],
        dtype=np.float64,
    )
    pred_grp_abs = pred_grp_raw * test_group_denoms
    pred_grp_rel = np.clip(pred_grp_abs / test_totals, 0.0, 1.0)

    # 策略 3：绝对剩余寿命（分钟）直接回归
    abs_model = RandomForestRULPredictor(
        feature_columns=cols,
        mode="absolute",
        n_estimators=n_estimators,
        random_state=random_state,
    ).fit(
        train_df,
        train_df["rul_minutes"],
        condition_avg_life=cond_avg_life,
        global_avg_life=global_avg_life,
    )
    pred_abs_min = abs_model.predict(test_df)
    pred_abs_rel = np.clip(pred_abs_min / test_totals, 0.0, 1.0)

    def _reg_metrics(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
        return {
            "mae": round(float(mean_absolute_error(yt, yp)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            "r2": round(float(r2_score(yt, yp)), 4) if len(yt) > 1 else 0.0,
        }

    return {
        "max_based": {
            "label": "Max-based 相对归一化 [0,1]",
            "relative_scale": _reg_metrics(y_test_rel, pred_max_rel),
            "minute_scale": _reg_metrics(y_test_abs, pred_max_abs),
            "predictions_relative": pred_max_rel.tolist(),
            "predictions_minutes": pred_max_abs.tolist(),
        },
        "group_minmax": {
            "label": "工况分组归一化",
            "native_scale": _reg_metrics(y_test_grp, pred_grp_raw),
            "relative_scale": _reg_metrics(y_test_rel, pred_grp_rel),
            "minute_scale": _reg_metrics(y_test_abs, pred_grp_abs),
            "predictions_relative": pred_grp_rel.tolist(),
            "predictions_minutes": pred_grp_abs.tolist(),
        },
        "absolute": {
            "label": "绝对剩余寿命 (分钟)",
            "relative_scale": _reg_metrics(y_test_rel, pred_abs_rel),
            "minute_scale": _reg_metrics(y_test_abs, pred_abs_min),
            "predictions_relative": pred_abs_rel.tolist(),
            "predictions_minutes": pred_abs_min.tolist(),
        },
    }
