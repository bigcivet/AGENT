"""视觉检测与分类子包。"""

from vision_agent.vision.classifier import (
    VISION_MODEL_LABELS,
    VISION_MODEL_REGISTRY,
    BaseVisionDefectClassifier,
    ClassificationResult,
    EfficientNetB0DefectClassifier,
    ResNet18DefectClassifier,
    create_vision_classifier,
    list_available_vision_models,
    load_vision_classifier,
)
from vision_agent.vision.dataset import BoundingBox, DefectSample, NEUDetCatalog, NEUDetDataset
from vision_agent.vision.yolo_detector import DetectionBox, YOLODefectDetector

__all__ = [
    "BoundingBox",
    "DefectSample",
    "NEUDetCatalog",
    "NEUDetDataset",
    "BaseVisionDefectClassifier",
    "ClassificationResult",
    "EfficientNetB0DefectClassifier",
    "ResNet18DefectClassifier",
    "VISION_MODEL_LABELS",
    "VISION_MODEL_REGISTRY",
    "create_vision_classifier",
    "list_available_vision_models",
    "load_vision_classifier",
    "DetectionBox",
    "YOLODefectDetector",
]
