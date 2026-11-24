"""异常检测、故障分类（RF / 1D-CNN / LSTM）与剩余寿命（RUL）预测模型子包。"""

from .anomaly import AnomalyResult, IsolationForestDetector
from .base import BaseAnomalyDetector, BaseFaultClassifier, BaseRULPredictor
from .classifier import FaultClassificationResult, RandomForestFaultClassifier
from .deep_classifiers import (
    CNN1DFaultClassifier,
    FAULT_CLASSIFIER_FILES,
    FAULT_CLASSIFIER_LABELS,
    LSTMFaultClassifier,
    load_fault_classifier,
)
from .rul import (
    RULPredictionResult,
    RandomForestRULPredictor,
    compare_rul_normalization_strategies,
    compute_group_normalized_rul,
)
from .trainer import ModelBundle, TrainingPipeline

__all__ = [
    "AnomalyResult",
    "BaseAnomalyDetector",
    "BaseFaultClassifier",
    "BaseRULPredictor",
    "CNN1DFaultClassifier",
    "FAULT_CLASSIFIER_FILES",
    "FAULT_CLASSIFIER_LABELS",
    "FaultClassificationResult",
    "IsolationForestDetector",
    "LSTMFaultClassifier",
    "ModelBundle",
    "RULPredictionResult",
    "RandomForestFaultClassifier",
    "RandomForestRULPredictor",
    "TrainingPipeline",
    "compare_rul_normalization_strategies",
    "compute_group_normalized_rul",
    "load_fault_classifier",
]
