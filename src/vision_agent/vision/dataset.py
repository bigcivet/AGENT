"""NEU-DET 金属表面缺陷数据集加载器与 VOC XML 解析模块。

兼容以下常见目录组织形式：
1. 标准 NEU-DET 检测版（含 train/validation 分割）：
   data/neu_det/
     ├── train/
     │   ├── images/<class_name>/*.jpg (或直接 *.jpg)
     │   └── annotations/*.xml
     └── validation/
         ├── images/<class_name>/*.jpg
         └── annotations/*.xml
2. 扁平检测版（IMAGES + ANNOTATIONS）：
   data/neu_det/
     ├── IMAGES/*.jpg
     └── ANNOTATIONS/*.xml
3. 分类目录版（按 6 个类别建子目录）：
   data/neu_det/
     ├── crazing/*.jpg
     ├── inclusion/*.jpg
     └── ...
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from vision_agent.config import DEFECT_CLASSES, normalize_defect_name

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class BoundingBox:
    """PASCAL VOC 边界框标注（像素坐标）。"""

    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class DefectSample:
    """单张 NEU-DET 样本元信息。"""

    image_path: Path
    label: str
    label_idx: int
    split_hint: str = "unspecified"  # "train" / "validation" / "unspecified"
    annotation_path: Path | None = None
    bboxes: list[BoundingBox] = field(default_factory=list)
    width: int = 200
    height: int = 200


def parse_voc_xml(xml_path: Path) -> tuple[int, int, list[BoundingBox]]:
    """解析 NEU-DET 的 PASCAL VOC XML 标注文件。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    width = int(size_node.findtext("width", "200")) if size_node is not None else 200
    height = int(size_node.findtext("height", "200")) if size_node is not None else 200

    bboxes: list[BoundingBox] = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name", "").strip()
        if not raw_name:
            continue
        label = normalize_defect_name(raw_name)
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = int(float(bnd.findtext("xmin", "0")))
        ymin = int(float(bnd.findtext("ymin", "0")))
        xmax = int(float(bnd.findtext("xmax", str(width))))
        ymax = int(float(bnd.findtext("ymax", str(height))))
        bboxes.append(BoundingBox(label=label, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))
    return width, height, bboxes


def infer_label_from_path(image_path: Path, bboxes: list[BoundingBox]) -> str | None:
    """按优先级推断样本主类别：VOC 标注 > 父目录名 > 文件名前缀。"""
    if bboxes:
        counts = Counter(b.label for b in bboxes if b.label in DEFECT_CLASSES)
        if counts:
            return counts.most_common(1)[0][0]

    parent_norm = normalize_defect_name(image_path.parent.name)
    if parent_norm in DEFECT_CLASSES:
        return parent_norm

    stem_lower = image_path.stem.lower()
    for cls in DEFECT_CLASSES:
        alt = cls.replace("-", "_")
        if stem_lower.startswith(cls) or stem_lower.startswith(alt):
            return cls
    return None


def build_image_transforms(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """构建适配 ResNet18 迁移学习的图像预处理与数据增强流水线。

    NEU-DET 原始图像为 200x200 灰度图，加载时会转为 RGB 3 通道以匹配 ResNet18 输入。
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


class NEUDetCatalog:
    """NEU-DET 数据集目录扫描器与训练/验证/测试划分管理器。"""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.class_to_idx: dict[str, int] = {cls: i for i, cls in enumerate(DEFECT_CLASSES)}
        self.idx_to_class: dict[int, str] = {i: cls for i, cls in enumerate(DEFECT_CLASSES)}
        self._samples: list[DefectSample] = []
        self.refresh()

    @property
    def is_available(self) -> bool:
        return len(self._samples) > 0

    @property
    def samples(self) -> list[DefectSample]:
        return list(self._samples)

    def _index_xml_files(self) -> dict[str, Path]:
        xml_map: dict[str, Path] = {}
        if not self.data_dir.exists():
            return xml_map
        for xml_file in self.data_dir.rglob("*.xml"):
            xml_map[xml_file.stem.lower()] = xml_file
        return xml_map

    def refresh(self) -> None:
        """重新扫描目录，加载所有图像样本及对应 VOC XML 标注。"""
        self._samples = []
        if not self.data_dir.exists():
            return

        xml_map = self._index_xml_files()
        image_files = sorted(
            p
            for p in self.data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for img_path in image_files:
            # 跳过自动导出的 yolo 子目录，避免重复计入
            if "yolo_dataset" in img_path.parts:
                continue

            xml_path = xml_map.get(img_path.stem.lower())
            width, height, bboxes = (200, 200, [])
            if xml_path is not None and xml_path.exists():
                try:
                    width, height, bboxes = parse_voc_xml(xml_path)
                except ET.ParseError:
                    bboxes = []

            label = infer_label_from_path(img_path, bboxes)
            if label is None or label not in self.class_to_idx:
                continue

            parts_lower = {part.lower() for part in img_path.parts}
            if "validation" in parts_lower or "val" in parts_lower or "test" in parts_lower:
                split_hint = "validation"
            elif "train" in parts_lower:
                split_hint = "train"
            else:
                split_hint = "unspecified"

            self._samples.append(
                DefectSample(
                    image_path=img_path,
                    label=label,
                    label_idx=self.class_to_idx[label],
                    split_hint=split_hint,
                    annotation_path=xml_path,
                    bboxes=bboxes,
                    width=width,
                    height=height,
                )
            )

    def summary(self) -> dict[str, object]:
        """返回数据集各类别数量、标注覆盖率等统计摘要。"""
        counts = Counter(s.label for s in self._samples)
        annotated = sum(1 for s in self._samples if s.bboxes)
        return {
            "data_dir": str(self.data_dir),
            "total_images": len(self._samples),
            "annotated_images": annotated,
            "class_counts": {cls: counts.get(cls, 0) for cls in DEFECT_CLASSES},
        }

    def split_samples(
        self,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> tuple[list[DefectSample], list[DefectSample]]:
        """返回 `(train_samples, val_samples)`。

        如果目录本身包含 `train/` 与 `validation/` 划分则直接尊重官方目录；
        否则按缺陷类别做分层随机划分。
        """
        explicit_train = [s for s in self._samples if s.split_hint == "train"]
        explicit_val = [s for s in self._samples if s.split_hint == "validation"]
        if explicit_train and explicit_val:
            return explicit_train, explicit_val

        labels = [s.label_idx for s in self._samples]
        min_count = min(Counter(labels).values()) if labels else 0
        stratify = labels if min_count >= 2 else None
        train_s, val_s = train_test_split(
            self._samples,
            test_size=val_ratio,
            random_state=seed,
            stratify=stratify,
        )
        return list(train_s), list(val_s)

    def split_train_val_test(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[DefectSample], list[DefectSample], list[DefectSample]]:
        """按 70% / 15% / 15%（可配置）对全部 NEU-DET 样本执行严格分层三划分。

        对齐语雀《NEU 金属表面瑕疵分类参考文档》，保证训练、验证、测试集在 6 类缺陷上
        分布均衡且样本完全无重叠。
        """
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio 必须等于 1.0，当前为 {total_ratio:.4f}"
            )
        if not self._samples:
            return [], [], []

        all_samples = list(self._samples)
        labels = [s.label_idx for s in all_samples]
        min_count = min(Counter(labels).values()) if labels else 0
        holdout_ratio = val_ratio + test_ratio
        stratify_1 = labels if min_count >= 3 else None

        train_s, temp_s = train_test_split(
            all_samples,
            test_size=holdout_ratio,
            random_state=seed,
            stratify=stratify_1,
        )

        temp_labels = [s.label_idx for s in temp_s]
        temp_min = min(Counter(temp_labels).values()) if temp_labels else 0
        relative_test_ratio = test_ratio / max(holdout_ratio, 1e-8)
        stratify_2 = temp_labels if temp_min >= 2 else None

        val_s, test_s = train_test_split(
            temp_s,
            test_size=relative_test_ratio,
            random_state=seed,
            stratify=stratify_2,
        )
        return list(train_s), list(val_s), list(test_s)

    def export_yolo_dataset(self, output_dir: str | Path, val_ratio: float = 0.2, seed: int = 42) -> Path:
        """将带 VOC XML 标注的 NEU-DET 数据导出为 Ultralytics YOLO 格式并生成 data.yaml。"""
        out_root = Path(output_dir)
        annotated_samples = [s for s in self._samples if s.bboxes]
        if not annotated_samples:
            raise ValueError("当前数据集中未找到有效的 VOC XML 边界框标注，无法导出 YOLO 数据集。")

        train_s, val_s = self.split_samples(val_ratio=val_ratio, seed=seed)
        train_s = [s for s in train_s if s.bboxes]
        val_s = [s for s in val_s if s.bboxes]

        for split_name, subset in (("train", train_s), ("val", val_s)):
            img_dir = out_root / "images" / split_name
            lbl_dir = out_root / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for sample in subset:
                dst_img = img_dir / sample.image_path.name
                shutil.copy2(sample.image_path, dst_img)

                lines: list[str] = []
                w = max(1, sample.width)
                h = max(1, sample.height)
                for box in sample.bboxes:
                    if box.label not in self.class_to_idx:
                        continue
                    cls_id = self.class_to_idx[box.label]
                    cx = ((box.xmin + box.xmax) / 2.0) / w
                    cy = ((box.ymin + box.ymax) / 2.0) / h
                    bw = max(1.0, float(box.xmax - box.xmin)) / w
                    bh = max(1.0, float(box.ymax - box.ymin)) / h
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                (lbl_dir / f"{sample.image_path.stem}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )

        yaml_lines = [
            f"path: {out_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(DEFECT_CLASSES)}",
            "names:",
        ]
        for idx, cls_name in enumerate(DEFECT_CLASSES):
            yaml_lines.append(f"  {idx}: {cls_name}")

        yaml_path = out_root / "data.yaml"
        yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
        return yaml_path


class NEUDetDataset(Dataset):
    """PyTorch Dataset：按需加载 200x200 灰度缺陷图并转为 RGB 张量。"""

    def __init__(
        self,
        samples: list[DefectSample],
        transform: Callable | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample.image_path) as img:
            rgb = img.convert("RGB")
        tensor = self.transform(rgb) if self.transform is not None else rgb
        return tensor, sample.label_idx
