"""视觉检测与分类子包。"""

from vision_agent.vision.classifier import ClassificationResult, ResNet18DefectClassifier
from vision_agent.vision.dataset import BoundingBox, DefectSample, NEUDetCatalog, NEUDetDataset
from vision_agent.vision.yolo_detector import DetectionBox, YOLODefectDetector

__all__ = [
    "BoundingBox",
    "DefectSample",
    "NEUDetCatalog",
    "NEUDetDataset",
    "ClassificationResult",
    "ResNet18DefectClassifier",
    "DetectionBox",
    "YOLODefectDetector",
]
