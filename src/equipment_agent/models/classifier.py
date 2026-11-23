"""基于随机森林（Random Forest）的轴承故障分类基线模型。

实现 `BaseFaultClassifier` 统一接口，后续可无缝替换为 CNN / 1D-ResNet 模型。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from equipment_agent.features.extractor import FEATURE_COLUMNS
from equipment_agent.models.base import BaseFaultClassifier


@dataclass
class FaultClassificationResult:
    """单个样本的故障分类结果。"""

    fault_type: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str = "RandomForestClassifier"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RandomForestFaultClassifier(BaseFaultClassifier):
    """随机森林轴承故障分类器（基线模型）。"""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        n_estimators: int = 200,
        max_depth: int | None = 16,
        random_state: int = 42,
    ) -> None:
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self.classes_: list[str] = []
        self._fitted: bool = False

    def _prepare_matrix(self, x: pd.DataFrame | np.ndarray | dict[str, float]) -> np.ndarray:
        if isinstance(x, dict):
            return np.array([[float(x[c]) for c in self.feature_columns]], dtype=np.float64)
        if isinstance(x, pd.DataFrame):
            return x[self.feature_columns].to_numpy(dtype=np.float64)
        arr = np.asarray(x, dtype=np.float64)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    def fit(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "RandomForestFaultClassifier":
        mat = self._prepare_matrix(x)
        labels = np.asarray(y, dtype=str)
        self.pipeline.fit(mat, labels)
        clf: RandomForestClassifier = self.pipeline.named_steps["clf"]
        self.classes_ = [str(c) for c in clf.classes_]
        self._fitted = True
        return self

    def predict(self, x: pd.DataFrame | np.ndarray | dict[str, float]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RandomForestFaultClassifier 尚未训练，请先调用 fit() 或 load()。")
        mat = self._prepare_matrix(x)
        return self.pipeline.predict(mat)

    def predict_proba(
        self, x: pd.DataFrame | np.ndarray | dict[str, float]
    ) -> list[dict[str, float]]:
        if not self._fitted:
            raise RuntimeError("RandomForestFaultClassifier 尚未训练，请先调用 fit() 或 load()。")
        mat = self._prepare_matrix(x)
        probas = self.pipeline.predict_proba(mat)
        results: list[dict[str, float]] = []
        for row in probas:
            results.append(
                {cls_name: round(float(p), 4) for cls_name, p in zip(self.classes_, row)}
            )
        return results

    def classify_one(
        self, features: dict[str, float] | pd.Series | np.ndarray
    ) -> FaultClassificationResult:
        """对单个采样文件执行故障分类，返回最优故障类型及置信度。"""
        if isinstance(features, pd.Series):
            features = features.to_dict()
        probs = self.predict_proba(features)[0]
        sorted_items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        best_fault, best_conf = sorted_items[0]
        return FaultClassificationResult(
            fault_type=best_fault,
            confidence=round(float(best_conf), 4),
            probabilities=probs,
        )

    def top_feature_importances(self, top_k: int = 10) -> list[tuple[str, float]]:
        """返回随机森林中最重要的前 K 个振动特征。"""
        if not self._fitted:
            return []
        clf: RandomForestClassifier = self.pipeline.named_steps["clf"]
        pairs = list(zip(self.feature_columns, clf.feature_importances_))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return [(name, round(float(val), 4)) for name, val in pairs[:top_k]]

    def evaluate(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> dict[str, Any]:
        y_true = np.asarray(y, dtype=str)
        y_pred = self.predict(x)
        labels = sorted(set(y_true) | set(y_pred))
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "report": classification_report(
                y_true, y_pred, labels=labels, output_dict=True, zero_division=0
            ),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_columns": self.feature_columns,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "pipeline": self.pipeline,
                "classes_": self.classes_,
                "fitted": self._fitted,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestFaultClassifier":
        payload = joblib.load(path)
        obj = cls(
            feature_columns=payload["feature_columns"],
            n_estimators=payload["n_estimators"],
            max_depth=payload["max_depth"],
            random_state=payload["random_state"],
        )
        obj.pipeline = payload["pipeline"]
        obj.classes_ = payload["classes_"]
        obj._fitted = payload["fitted"]
        return obj
