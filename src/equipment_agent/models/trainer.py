"""模型统一训练、评估与持久化流水线。

将特征提取、孤立森林异常检测、随机森林故障分类与 RUL 预测串联成一键训练入口，
并按“轴承单元”进行训练/测试切分，避免同一轴承数据跨集泄露。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equipment_agent.data.loader import XJTUSYDataLoader
from equipment_agent.features.extractor import FEATURE_COLUMNS, build_feature_dataset
from equipment_agent.models.anomaly import IsolationForestDetector
from equipment_agent.models.base import BaseFaultClassifier
from equipment_agent.models.classifier import RandomForestFaultClassifier
from equipment_agent.models.deep_classifiers import (
    CNN1DFaultClassifier,
    FAULT_CLASSIFIER_FILES,
    LSTMFaultClassifier,
)
from equipment_agent.models.rul import RandomForestRULPredictor


ANOMALY_MODEL_FILE = "isolation_forest.joblib"
CLASSIFIER_MODEL_FILE = "rf_fault_classifier.joblib"
CNN1D_CLASSIFIER_FILE = FAULT_CLASSIFIER_FILES["cnn1d"]
LSTM_CLASSIFIER_FILE = FAULT_CLASSIFIER_FILES["lstm"]
RUL_MODEL_FILE = "rf_rul_predictor.joblib"
METRICS_FILE = "metrics.json"


@dataclass
class ModelBundle:
    """已加载的完整诊断模型集合（支持在 RF / 1D-CNN / LSTM 故障分类器间动态切换）。"""

    anomaly_detector: IsolationForestDetector
    fault_classifier: BaseFaultClassifier
    rul_predictor: RandomForestRULPredictor
    metrics: dict[str, Any]
    classifiers: dict[str, BaseFaultClassifier] = field(default_factory=dict)
    active_classifier_key: str = "rf"

    def __post_init__(self) -> None:
        if not self.classifiers:
            self.classifiers = {self.active_classifier_key: self.fault_classifier}
        elif self.active_classifier_key not in self.classifiers:
            self.classifiers[self.active_classifier_key] = self.fault_classifier

    def available_classifiers(self) -> list[str]:
        """返回当前模型束中已加载可用的故障分类器键列表（按 rf/cnn1d/lstm 顺序）。"""
        order = ("rf", "cnn1d", "lstm")
        keys = [k for k in order if k in self.classifiers]
        for k in self.classifiers:
            if k not in keys:
                keys.append(k)
        return keys

    def set_active_classifier(self, model_key: str) -> BaseFaultClassifier:
        """动态切换当前激活的故障分类模型（`rf` / `cnn1d` / `lstm`）。"""
        key = model_key.strip().lower()
        if key not in self.classifiers:
            raise KeyError(
                f"故障分类模型 `{model_key}` 尚未加载，当前可用：{self.available_classifiers()}"
            )
        self.active_classifier_key = key
        self.fault_classifier = self.classifiers[key]
        return self.fault_classifier

    @classmethod
    def exists(cls, artifact_dir: str | Path) -> bool:
        root = Path(artifact_dir)
        return all(
            (root / fname).exists()
            for fname in (ANOMALY_MODEL_FILE, CLASSIFIER_MODEL_FILE, RUL_MODEL_FILE)
        )

    @classmethod
    def load(cls, artifact_dir: str | Path, active_classifier: str = "rf") -> "ModelBundle":
        root = Path(artifact_dir)
        if not cls.exists(root):
            raise FileNotFoundError(
                f"模型文件不完整（目录：{root}），请先运行 `python scripts/train_pipeline.py` 完成训练。"
            )
        metrics_path = root / METRICS_FILE
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {}
        )
        rf_clf = RandomForestFaultClassifier.load(root / CLASSIFIER_MODEL_FILE)
        classifiers: dict[str, BaseFaultClassifier] = {"rf": rf_clf}

        cnn_path = root / CNN1D_CLASSIFIER_FILE
        if cnn_path.exists():
            classifiers["cnn1d"] = CNN1DFaultClassifier.load(cnn_path)

        lstm_path = root / LSTM_CLASSIFIER_FILE
        if lstm_path.exists():
            classifiers["lstm"] = LSTMFaultClassifier.load(lstm_path)

        chosen_key = active_classifier.strip().lower()
        if chosen_key not in classifiers:
            chosen_key = "rf"

        return cls(
            anomaly_detector=IsolationForestDetector.load(root / ANOMALY_MODEL_FILE),
            fault_classifier=classifiers[chosen_key],
            rul_predictor=RandomForestRULPredictor.load(root / RUL_MODEL_FILE),
            metrics=metrics,
            classifiers=classifiers,
            active_classifier_key=chosen_key,
        )

    def save(self, artifact_dir: str | Path) -> Path:
        root = Path(artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.anomaly_detector.save(root / ANOMALY_MODEL_FILE)
        # 保存全部已训练的故障分类模型（RF / 1D-CNN / LSTM）
        rf_model = self.classifiers.get("rf", self.fault_classifier)
        rf_model.save(root / CLASSIFIER_MODEL_FILE)
        if "cnn1d" in self.classifiers:
            self.classifiers["cnn1d"].save(root / CNN1D_CLASSIFIER_FILE)
        if "lstm" in self.classifiers:
            self.classifiers["lstm"].save(root / LSTM_CLASSIFIER_FILE)
        self.rul_predictor.save(root / RUL_MODEL_FILE)
        (root / METRICS_FILE).write_text(
            json.dumps(self.metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return root


class TrainingPipeline:
    """设备运维诊断智能体的离线训练与评估流水线。"""

    def __init__(
        self,
        loader: XJTUSYDataLoader,
        artifact_dir: str | Path,
        healthy_life_cutoff: float = 0.25,
        fault_train_window: tuple[float, float] = (0.30, 0.90),
        test_ratio: float = 0.30,
        step: int = 1,
        random_state: int = 42,
        train_deep_models: bool = False,
        deep_epochs: int = 35,
        deep_batch_size: int = 32,
    ) -> None:
        self.loader = loader
        self.artifact_dir = Path(artifact_dir)
        self.healthy_life_cutoff = healthy_life_cutoff
        self.fault_train_window = fault_train_window
        self.test_ratio = test_ratio
        self.step = step
        self.random_state = random_state
        self.train_deep_models = train_deep_models
        self.deep_epochs = deep_epochs
        self.deep_batch_size = deep_batch_size

    def run(self) -> ModelBundle:
        """执行完整训练流程并保存模型与评估报告。"""
        if not self.loader.is_available:
            raise RuntimeError(
                f"数据目录 `{self.loader.data_dir}` 下未找到任何 XJTU-SY 轴承 CSV 数据。"
            )

        train_bearings, test_bearings = self.loader.split_bearings_by_unit(
            test_ratio=self.test_ratio, seed=self.random_state
        )

        train_df = build_feature_dataset(self.loader, bearings=train_bearings, step=self.step)
        test_df = (
            build_feature_dataset(self.loader, bearings=test_bearings, step=self.step)
            if test_bearings
            else pd.DataFrame()
        )

        # 1. 训练无监督异常检测模型（仅使用训练轴承的早期健康阶段）
        healthy_df = train_df[train_df["life_ratio"] <= self.healthy_life_cutoff]
        if len(healthy_df) < 10:
            healthy_df = train_df.nsmallest(max(10, len(train_df) // 4), "life_ratio")

        anomaly_detector = IsolationForestDetector(
            feature_columns=FEATURE_COLUMNS,
            contamination=0.05,
            n_estimators=200,
            random_state=self.random_state,
        ).fit(healthy_df)

        # 评估异常检测：早期健康段误报率 vs 晚期退化段检出率
        eval_df = test_df if not test_df.empty else train_df
        eval_early = eval_df[eval_df["life_ratio"] <= self.healthy_life_cutoff]
        eval_late = eval_df[eval_df["life_ratio"] >= 0.70]
        early_alarm_rate = (
            float(np.mean(anomaly_detector.predict(eval_early))) if not eval_early.empty else 0.0
        )
        late_detect_rate = (
            float(np.mean(anomaly_detector.predict(eval_late))) if not eval_late.empty else 0.0
        )

        # 2. 训练故障分类模型（在显著退化阶段区分故障类型，并保留早期“正常”类别便于完整识别）
        low_w, high_w = self.fault_train_window
        clf_train_fault = train_df[
            (train_df["life_ratio"] >= low_w) & (train_df["life_ratio"] <= high_w)
        ].copy()
        if clf_train_fault.empty:
            clf_train_fault = train_df.copy()

        clf_train_normal = train_df[train_df["life_ratio"] <= min(0.20, self.healthy_life_cutoff)].copy()
        if not clf_train_normal.empty:
            clf_train_normal["fault_type"] = "正常"
            clf_train_all = pd.concat([clf_train_fault, clf_train_normal], ignore_index=True)
        else:
            clf_train_all = clf_train_fault

        fault_classifier = RandomForestFaultClassifier(
            feature_columns=FEATURE_COLUMNS,
            n_estimators=200,
            random_state=self.random_state,
        ).fit(clf_train_all, clf_train_all["fault_type"])

        classifiers: dict[str, BaseFaultClassifier] = {"rf": fault_classifier}

        # 故障分类测试评估（聚焦退化阶段的中后期样本，与课程实训思路一致）
        if not test_df.empty:
            clf_test_df = test_df[
                (test_df["life_ratio"] >= low_w) & (test_df["life_ratio"] <= high_w)
            ]
            if clf_test_df.empty:
                clf_test_df = test_df
            clf_metrics = fault_classifier.evaluate(clf_test_df, clf_test_df["fault_type"])
        else:
            clf_test_df = clf_train_all
            clf_metrics = fault_classifier.evaluate(clf_train_all, clf_train_all["fault_type"])

        deep_clf_metrics: dict[str, Any] = {}
        if self.train_deep_models:
            cnn_clf = CNN1DFaultClassifier(
                feature_columns=FEATURE_COLUMNS,
                epochs=self.deep_epochs,
                batch_size=self.deep_batch_size,
                random_state=self.random_state,
            ).fit(clf_train_all, clf_train_all["fault_type"])
            classifiers["cnn1d"] = cnn_clf
            deep_clf_metrics["cnn1d"] = cnn_clf.evaluate(clf_test_df, clf_test_df["fault_type"])

            lstm_clf = LSTMFaultClassifier(
                feature_columns=FEATURE_COLUMNS,
                epochs=self.deep_epochs,
                batch_size=self.deep_batch_size,
                random_state=self.random_state,
            ).fit(clf_train_all, clf_train_all["fault_type"])
            classifiers["lstm"] = lstm_clf
            deep_clf_metrics["lstm"] = lstm_clf.evaluate(clf_test_df, clf_test_df["fault_type"])

        # 3. 训练 RUL 剩余寿命预测模型（采用 Max-based 归一化相对寿命，并记录工况参考平均寿命）
        cond_avg_life = (
            train_df.groupby("condition")["total_files"].mean().to_dict()
            if not train_df.empty
            else {}
        )
        global_avg_life = float(train_df["total_files"].mean()) if not train_df.empty else 120.0

        rul_predictor = RandomForestRULPredictor(
            feature_columns=FEATURE_COLUMNS,
            mode="relative",
            n_estimators=200,
            random_state=self.random_state,
        ).fit(
            train_df,
            train_df["rul_normalized"],
            condition_avg_life=cond_avg_life,
            global_avg_life=global_avg_life,
        )

        rul_eval_df = test_df if not test_df.empty else train_df
        rul_metrics = rul_predictor.evaluate(rul_eval_df, rul_eval_df["rul_normalized"])

        metrics: dict[str, Any] = {
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_dir": str(self.loader.data_dir),
            "train_bearings": [b.key for b in train_bearings],
            "test_bearings": [b.key for b in test_bearings],
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "anomaly_detection": {
                "healthy_life_cutoff": self.healthy_life_cutoff,
                "early_false_alarm_rate": round(early_alarm_rate, 4),
                "late_stage_detection_rate": round(late_detect_rate, 4),
            },
            "fault_classification": {
                "accuracy": clf_metrics["accuracy"],
                "macro_f1": clf_metrics["macro_f1"],
                "labels": clf_metrics["labels"],
                "confusion_matrix": clf_metrics["confusion_matrix"],
                "top_features": fault_classifier.top_feature_importances(top_k=8),
            },
            "rul_prediction": rul_metrics,
        }

        if deep_clf_metrics:
            metrics["deep_fault_classifiers"] = deep_clf_metrics

        bundle = ModelBundle(
            anomaly_detector=anomaly_detector,
            fault_classifier=fault_classifier,
            rul_predictor=rul_predictor,
            metrics=metrics,
            classifiers=classifiers,
            active_classifier_key="rf",
        )
        bundle.save(self.artifact_dir)
        return bundle
