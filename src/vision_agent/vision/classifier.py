"""基于 PyTorch ResNet18 迁移学习的金属表面缺陷分类模型。

设计要点：
1. 默认尝试加载 ImageNet 预训练权重进行迁移学习；若离线环境无法下载权重，
   自动安全回退为随机初始化并继续训练，保证实训环境 100% 可跑通。
2. 支持 CPU / GPU 自动适配，训练轮数（epochs）、批次大小（batch_size）、
   是否冻结主干（freeze_backbone）均可配置。
3. 训练结束后自动输出并保存准确率（Accuracy）、宏平均 F1（Macro-F1）、
   分类明细及 6x6 混淆矩阵。
"""

from __future__ import annotations

import io
import json
import random
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
from vision_agent.vision.dataset import NEUDetCatalog, NEUDetDataset, build_image_transforms


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
            # 离线环境下载预训练权重失败时自动降级，避免阻断训练
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model, used_pretrained


class ResNet18DefectClassifier:
    """NEU-DET 六类金属表面缺陷 ResNet18 迁移学习分类器。"""

    MODEL_FILENAME = "resnet18_neu_det.pt"
    METRICS_FILENAME = "vision_metrics.json"

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
    def exists(cls, artifact_dir: str | Path) -> bool:
        return (Path(artifact_dir) / cls.MODEL_FILENAME).exists()

    def fit(
        self,
        catalog: NEUDetCatalog,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 1e-3,
        val_ratio: float = 0.2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        seed: int = 42,
    ) -> dict[str, Any]:
        """在 NEU-DET 数据集上执行迁移学习训练并返回完整评估指标。"""
        if not catalog.is_available:
            raise ValueError(f"未在目录 {catalog.data_dir} 扫描到有效的 NEU-DET 图像样本。")

        _set_seed(seed)
        train_samples, val_samples = catalog.split_samples(val_ratio=val_ratio, seed=seed)

        train_ds = NEUDetDataset(
            train_samples,
            transform=build_image_transforms(image_size=self.image_size, is_train=True),
        )
        val_ds = NEUDetDataset(
            val_samples,
            transform=build_image_transforms(image_size=self.image_size, is_train=False),
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model, used_pretrained = _build_resnet18(
            num_classes=len(self.classes), pretrained=pretrained
        )
        self.used_pretrained = used_pretrained

        # 若选择冻结主干，仅训练最后的 layer4 + fc（在 CPU 上兼顾速度与精度）
        if freeze_backbone and used_pretrained:
            for name, param in model.named_parameters():
                if not (name.startswith("fc.") or name.startswith("layer4.")):
                    param.requires_grad = False

        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        history: list[dict[str, float]] = []
        best_val_acc = -1.0
        best_state: dict[str, torch.Tensor] | None = None

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

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        self.model.eval()

        final_eval = self._evaluate_loader(self.model, val_loader, criterion)
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
            "model_name": "resnet18",
            "device": str(self.device),
            "pretrained_weights_used": used_pretrained,
            "freeze_backbone": freeze_backbone,
            "image_size": self.image_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "classes": list(self.classes),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
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
                "state_dict": self.model.cpu().state_dict(),
                "classes": list(self.classes),
                "image_size": self.image_size,
                "used_pretrained": self.used_pretrained,
            },
            ckpt_path,
        )
        # 恢复到推理设备
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
    ) -> "ResNet18DefectClassifier":
        """从磁盘加载已训练的 ResNet18 缺陷分类器与评估指标。"""
        art_dir = Path(artifact_dir)
        ckpt_path = art_dir / cls.MODEL_FILENAME
        if not ckpt_path.exists():
            raise FileNotFoundError(f"未找到视觉分类模型文件：{ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        classes = tuple(checkpoint.get("classes", DEFECT_CLASSES))
        image_size = int(checkpoint.get("image_size", 224))

        instance = cls(classes=classes, image_size=image_size, device=device)
        model, _ = _build_resnet18(num_classes=len(classes), pretrained=False)
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
            model_name="resnet18",
        )
