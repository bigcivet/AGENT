"""可选 YOLO（ultralytics）目标检测适配器。

说明：
- 默认主流程使用轻量、CPU 友好的 PyTorch ResNet18 迁移学习分类器；
- 若环境中已安装 `ultralytics`，可直接通过本模块基于 NEU-DET 的 VOC XML 标注
  一键导出 YOLO 数据集、训练 YOLOv8n 检测模型并执行带检测框的推理。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vision_agent.config import DEFECT_CLASSES, DEFECT_CN_MAP, normalize_defect_name
from vision_agent.vision.dataset import NEUDetCatalog


@dataclass
class DetectionBox:
    """单个目标检测框结果。"""

    defect_type: str
    defect_type_cn: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class YOLODefectDetector:
    """Ultralytics YOLO 检测适配器（按需启用）。"""

    def __init__(self, weights_path: str | Path = "yolov8n.pt") -> None:
        self.weights_path = str(weights_path)
        self._model: Any = None

    @staticmethod
    def is_ultralytics_installed() -> bool:
        try:
            import ultralytics  # noqa: F401

            return True
        except ImportError:
            return False

    def _ensure_available(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "当前环境未安装 ultralytics，无法启用 YOLO 目标检测。"
                "如需使用，请执行：pip install ultralytics"
            ) from exc
        return YOLO

    def train_from_catalog(
        self,
        catalog: NEUDetCatalog,
        work_dir: str | Path,
        epochs: int = 10,
        imgsz: int = 224,
        batch: int = 16,
    ) -> Path:
        """将 NEU-DET VOC 标注导出为 YOLO 格式并微调 YOLO 检测模型。"""
        YOLO = self._ensure_available()
        work_root = Path(work_dir)
        yolo_data_dir = work_root / "yolo_dataset"
        yaml_path = catalog.export_yolo_dataset(yolo_data_dir)

        model = YOLO(self.weights_path)
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(work_root / "runs"),
            name="neu_det_yolo",
            exist_ok=True,
        )
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        self.weights_path = str(best_pt)
        self._model = YOLO(self.weights_path)
        return best_pt

    def predict(self, image_path: str | Path, conf: float = 0.25) -> list[DetectionBox]:
        """对单张图像执行目标检测，返回检测框列表。"""
        YOLO = self._ensure_available()
        if self._model is None:
            self._model = YOLO(self.weights_path)

        preds = self._model.predict(source=str(image_path), conf=conf, verbose=False)
        boxes_out: list[DetectionBox] = []
        if not preds:
            return boxes_out

        res = preds[0]
        names = res.names or {i: cls for i, cls in enumerate(DEFECT_CLASSES)}
        for b in res.boxes:
            cls_id = int(b.cls.item())
            raw_name = str(names.get(cls_id, DEFECT_CLASSES[cls_id % len(DEFECT_CLASSES)]))
            norm_name = normalize_defect_name(raw_name)
            xyxy = [round(float(v), 1) for v in b.xyxy[0].tolist()]
            boxes_out.append(
                DetectionBox(
                    defect_type=norm_name,
                    defect_type_cn=DEFECT_CN_MAP.get(norm_name, norm_name),
                    confidence=round(float(b.conf.item()), 4),
                    bbox=xyxy,
                )
            )
        return boxes_out
