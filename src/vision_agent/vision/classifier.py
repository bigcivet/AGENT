"""基于 PyTorch 迁移学习的金属表面缺陷多模型分类模块（ResNet18 / EfficientNet-B0）。

设计要点：
1. 统一封装 `BaseVisionDefectClassifier`，提供 `ResNet18DefectClassifier` 与
   `EfficientNetB0DefectClassifier` 两个可互换分类模型；
2. 默认尝试加载 ImageNet 预训练权重；若离线环境或网络受限无法下载权重，
   自动安全回退为 `weights=None` 随机初始化并继续训练，保证 100% 可跑通；
3. 支持两划分（train/val）与标准三划分（70% train / 15% val / 15% test），
   训练后自动统计 Accuracy、Macro-F1、每类 Precision/Recall/F1、6×6 混淆矩阵、
   训练耗时与单图平均推理耗时（ms）。
"""

from __future__ import annotations

import io
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import models

from vision_agent.config import DEFECT_CLASSES, DEFECT_CN_MAP, DEFECT_SEVERITY_MAP
from vision_agent.vision.dataset import (
    DefectSample,
    NEUDetCatalog,
    NEUDetDataset,
    build_image_transforms,
)


@dataclass
class ClassificationResult:
    """单张金属表面图像的缺陷分类推理结果。"""

    defect_type: str
    defect_type_cn: str
    confidence: float
    severity: str
    probabilities: dict[str, float]
    model_name: str = "resnet18"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_resnet18(num_classes: int, pretrained: bool = True) -> tuple[nn.Module, bool]:
    """构建 ResNet18 模型，返回 `(model, used_pretrained)`。"""
    used_pretrained = False
    model: nn.Module
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            used_pretrained = True
        except Exception:
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model, used_pretrained


def _build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> tuple[nn.Module, bool]:
    """构建 EfficientNet-B0 模型，离线或下载失败时自动回退 `weights=None`。"""
    used_pretrained = False
    model: nn.Module
    if pretrained:
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
            used_pretrained = True
        except Exception:
            model = models.efficientnet_b0(weights=None)
    else:
        model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model, used_pretrained


class BaseVisionDefectClassifier:
    """NEU-DET 六类金属表面缺陷迁移学习分类器共享基类。"""

    ARCH_NAME: str = "resnet18"
    MODEL_FILENAME: str = "resnet18_neu_det.pt"
    METRICS_FILENAME: str = "vision_metrics.json"

    def __init__(
        self,
        classes: tuple[str, ...] = DEFECT_CLASSES,
        image_size: int = 224,
        device: str | None = None,
    ) -> None:
        self.classes = tuple(classes)
        self.image_size = image_size
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: nn.Module | None = None
        self.metrics: dict[str, Any] = {}
        self.used_pretrained: bool = False
        self._eval_transform = build_image_transforms(image_size=self.image_size, is_train=False)

    @classmethod
    def _build_backbone(cls, num_classes: int, pretrained: bool = True) -> tuple[nn.Module, bool]:
        raise NotImplementedError

    @classmethod
    def _apply_freeze_backbone(cls, model: nn.Module) -> None:
        """冻结主干浅层参数，仅保留高层语义块与分类头可训练。"""
        raise NotImplementedError

    @classmethod
    def exists(cls, artifact_dir: str | Path) -> bool:
        return (Path(artifact_dir) / cls.MODEL_FILENAME).exists()

    def fit(
        self,
        catalog: NEUDetCatalog | None = None,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 1e-3,
        val_ratio: float = 0.2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        seed: int = 42,
        split_mode: str = "two_way",
        train_ratio: float = 0.70,
        test_ratio: float = 0.15,
        custom_splits: tuple[list[DefectSample], list[DefectSample], list[DefectSample] | None] | None = None,
    ) -> dict[str, Any]:
        """执行训练与评估并返回完整指标字典。

        Args:
            catalog: NEU-DET 目录管理器（当 `custom_splits` 为 None 时必填）。
            split_mode: `"two_way"`（兼容旧版 train/val）或 `"three_way"`（70%/15%/15%）。
            custom_splits: 显式传入 `(train_samples, val_samples, test_samples)`，确保多模型在完全相同数据划分上对比。
        """
        _set_seed(seed)

        if custom_splits is not None:
            train_samples, val_samples, test_samples = custom_splits
            effective_split_mode = "three_way" if test_samples is not None else "two_way"
        else:
            if catalog is None or not catalog.is_available:
                raise ValueError("未扫描到有效的 NEU-DET 图像样本，无法训练视觉模型。")
            if split_mode == "three_way":
                effective_val_ratio = round(1.0 - train_ratio - test_ratio, 6)
                train_samples, val_samples, test_samples = catalog.split_train_val_test(
                    train_ratio=train_ratio,
                    val_ratio=effective_val_ratio,
                    test_ratio=test_ratio,
                    seed=seed,
                )
                effective_split_mode = "three_way"
            else:
                train_samples, val_samples = catalog.split_samples(val_ratio=val_ratio, seed=seed)
                test_samples = None
                effective_split_mode = "two_way"

        train_ds = NEUDetDataset(
            train_samples,
            transform=build_image_transforms(image_size=self.image_size, is_train=True),
        )
        val_ds = NEUDetDataset(
            val_samples,
            transform=build_image_transforms(image_size=self.image_size, is_train=False),
        )
        test_ds = (
            NEUDetDataset(
                test_samples,
                transform=build_image_transforms(image_size=self.image_size, is_train=False),
            )
            if test_samples is not None
            else None
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = (
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            if test_ds is not None
            else None
        )

        model, used_pretrained = self._build_backbone(
            num_classes=len(self.classes), pretrained=pretrained
        )
        self.used_pretrained = used_pretrained

        if freeze_backbone and used_pretrained:
            self._apply_freeze_backbone(model)

        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_param_count = int(sum(p.numel() for p in model.parameters()))
        trainable_param_count = int(sum(p.numel() for p in trainable_params))

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        history: list[dict[str, float]] = []
        best_val_acc = -1.0
        best_state: dict[str, torch.Tensor] | None = None

        train_t0 = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, targets in train_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * images.size(0)
                preds = logits.argmax(dim=1)
                train_correct += int((preds == targets).sum().item())
                train_total += int(targets.size(0))

            scheduler.step()

            val_eval = self._evaluate_loader(model, val_loader, criterion)
            train_loss = running_loss / max(1, train_total)
            train_acc = train_correct / max(1, train_total)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(val_eval["loss"], 4),
                    "val_acc": round(val_eval["accuracy"], 4),
                }
            )

            if val_eval["accuracy"] >= best_val_acc:
                best_val_acc = val_eval["accuracy"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        train_elapsed = time.perf_counter() - train_t0

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        self.model.eval()

        # 若存在独立测试集则在测试集上给出最终指标，否则在验证集上评估
        eval_loader = test_loader if test_loader is not None else val_loader
        eval_split_name = "test" if test_loader is not None else "validation"

        infer_t0 = time.perf_counter()
        final_eval = self._evaluate_loader(self.model, eval_loader, criterion)
        infer_elapsed = time.perf_counter() - infer_t0
        n_eval = max(1, len(final_eval["y_true"]))

        y_true = final_eval["y_true"]
        y_pred = final_eval["y_pred"]

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.classes))))
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(self.classes))),
            target_names=list(self.classes),
            output_dict=True,
            zero_division=0,
        )

        self.metrics = {
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.ARCH_NAME,
            "device": str(self.device),
            "split_mode": effective_split_mode,
            "eval_split": eval_split_name,
            "pretrained_weights_used": used_pretrained,
            "freeze_backbone": freeze_backbone,
            "image_size": self.image_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "total_params": total_param_count,
            "trainable_params": trainable_param_count,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples) if test_samples is not None else 0,
            "best_val_accuracy": round(float(max(best_val_acc, 0.0)), 4),
            "classes": list(self.classes),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "train_time_seconds": round(float(train_elapsed), 3),
            "eval_infer_time_seconds": round(float(infer_elapsed), 4),
            "avg_infer_time_ms": round((infer_elapsed / n_eval) * 1000.0, 3),
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": {
                cls: {
                    "cn_name": DEFECT_CN_MAP.get(cls, cls),
                    "precision": round(float(report[cls]["precision"]), 4),
                    "recall": round(float(report[cls]["recall"]), 4),
                    "f1_score": round(float(report[cls]["f1-score"]), 4),
                    "support": int(report[cls]["support"]),
                }
                for cls in self.classes
                if cls in report
            },
            "history": history,
        }
        return self.metrics

    @torch.no_grad()
    def _evaluate_loader(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> dict[str, Any]:
        model.eval()
        total_loss = 0.0
        total = 0
        y_true: list[int] = []
        y_pred: list[int] = []

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * images.size(0)
            total += int(targets.size(0))
            preds = logits.argmax(dim=1)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        return {
            "loss": total_loss / max(1, total),
            "accuracy": float(acc),
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def save(self, artifact_dir: str | Path) -> Path:
        """持久化保存 PyTorch 权重文件与 JSON 评估报告。"""
        if self.model is None:
            raise RuntimeError("模型尚未训练或加载，无法保存。")
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = out_dir / self.MODEL_FILENAME
        torch.save(
            {
                "arch_name": self.ARCH_NAME,
                "state_dict": self.model.cpu().state_dict(),
                "classes": list(self.classes),
                "image_size": self.image_size,
                "used_pretrained": self.used_pretrained,
            },
            ckpt_path,
        )
        self.model.to(self.device)

        metrics_path = out_dir / self.METRICS_FILENAME
        metrics_path.write_text(
            json.dumps(self.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return ckpt_path

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        device: str | None = None,
    ) -> "BaseVisionDefectClassifier":
        """从磁盘加载已训练的视觉缺陷分类器与评估指标。"""
        art_dir = Path(artifact_dir)
        ckpt_path = art_dir / cls.MODEL_FILENAME
        if not ckpt_path.exists():
            raise FileNotFoundError(f"未找到视觉分类模型文件：{ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        classes = tuple(checkpoint.get("classes", DEFECT_CLASSES))
        image_size = int(checkpoint.get("image_size", 224))

        instance = cls(classes=classes, image_size=image_size, device=device)
        model, _ = cls._build_backbone(num_classes=len(classes), pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(instance.device)
        model.eval()
        instance.model = model
        instance.used_pretrained = bool(checkpoint.get("used_pretrained", False))

        metrics_path = art_dir / cls.METRICS_FILENAME
        if metrics_path.exists():
            instance.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return instance

    @torch.no_grad()
    def predict_image(
        self, image_source: str | Path | bytes | Image.Image
    ) -> ClassificationResult:
        """对单张图像（路径 / bytes / PIL.Image）执行表面缺陷分类推理。"""
        if self.model is None:
            raise RuntimeError("模型尚未训练或加载，请先调用 fit() 或 load()。")

        if isinstance(image_source, (str, Path)):
            with Image.open(image_source) as img:
                rgb = img.convert("RGB")
        elif isinstance(image_source, bytes):
            with Image.open(io.BytesIO(image_source)) as img:
                rgb = img.convert("RGB")
        elif isinstance(image_source, Image.Image):
            rgb = image_source.convert("RGB")
        else:
            raise TypeError(f"不支持的图像输入类型：{type(image_source)}")

        tensor = self._eval_transform(rgb).unsqueeze(0).to(self.device)
        self.model.eval()
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_idx = int(np.argmax(probs))
        defect_type = self.classes[top_idx]
        confidence = float(probs[top_idx])

        prob_dict = {
            cls_name: round(float(probs[i]), 4) for i, cls_name in enumerate(self.classes)
        }
        return ClassificationResult(
            defect_type=defect_type,
            defect_type_cn=DEFECT_CN_MAP.get(defect_type, defect_type),
            confidence=round(confidence, 4),
            severity=DEFECT_SEVERITY_MAP.get(defect_type, "中"),
            probabilities=prob_dict,
            model_name=self.ARCH_NAME,
        )


class ResNet18DefectClassifier(BaseVisionDefectClassifier):
    """NEU-DET 六类金属表面缺陷 ResNet18 迁移学习分类器。"""

    ARCH_NAME = "resnet18"
    MODEL_FILENAME = "resnet18_neu_det.pt"
    METRICS_FILENAME = "vision_metrics.json"

    @classmethod
    def _build_backbone(cls, num_classes: int, pretrained: bool = True) -> tuple[nn.Module, bool]:
        return _build_resnet18(num_classes=num_classes, pretrained=pretrained)

    @classmethod
    def _apply_freeze_backbone(cls, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not (name.startswith("fc.") or name.startswith("layer4.")):
                param.requires_grad = False


class EfficientNetB0DefectClassifier(BaseVisionDefectClassifier):
    """NEU-DET 六类金属表面缺陷 EfficientNet-B0 轻量高效分类器。"""

    ARCH_NAME = "efficientnet_b0"
    MODEL_FILENAME = "efficientnet_b0_neu_det.pt"
    METRICS_FILENAME = "efficientnet_b0_metrics.json"

    @classmethod
    def _build_backbone(cls, num_classes: int, pretrained: bool = True) -> tuple[nn.Module, bool]:
        return _build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)

    @classmethod
    def _apply_freeze_backbone(cls, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not (
                name.startswith("classifier.")
                or name.startswith("features.7.")
                or name.startswith("features.8.")
            ):
                param.requires_grad = False


VISION_MODEL_REGISTRY: dict[str, type[BaseVisionDefectClassifier]] = {
    "resnet18": ResNet18DefectClassifier,
    "efficientnet_b0": EfficientNetB0DefectClassifier,
}

VISION_MODEL_LABELS: dict[str, str] = {
    "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNet-B0",
}


def list_available_vision_models(artifact_dir: str | Path) -> list[str]:
    """扫描视觉模型目录下已训练就绪的模型标识列表。"""
    return [name for name, cls in VISION_MODEL_REGISTRY.items() if cls.exists(artifact_dir)]


def create_vision_classifier(
    model_name: str,
    image_size: int = 224,
    device: str | None = None,
) -> BaseVisionDefectClassifier:
    """按模型名实例化未训练的视觉分类器。"""
    key = model_name.strip().lower()
    if key not in VISION_MODEL_REGISTRY:
        raise ValueError(
            f"不支持的视觉分类模型：{model_name}，可选：{list(VISION_MODEL_REGISTRY.keys())}"
        )
    return VISION_MODEL_REGISTRY[key](image_size=image_size, device=device)


def load_vision_classifier(
    model_name: str,
    artifact_dir: str | Path,
    device: str | None = None,
) -> BaseVisionDefectClassifier:
    """按模型名从磁盘加载对应视觉分类器。"""
    key = model_name.strip().lower()
    if key not in VISION_MODEL_REGISTRY:
        raise ValueError(
            f"不支持的视觉分类模型：{model_name}，可选：{list(VISION_MODEL_REGISTRY.keys())}"
        )
    return VISION_MODEL_REGISTRY[key].load(artifact_dir=artifact_dir, device=device)
