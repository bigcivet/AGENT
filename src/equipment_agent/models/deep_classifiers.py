"""基于 PyTorch 的 1D-CNN 与 LSTM 轴承故障分类器实现。

严格继承 `BaseFaultClassifier` 统一接口，与 `RandomForestFaultClassifier` 完全互换：
- `CNN1DFaultClassifier`：将标准化后的 54 维时频特征序列送入多层一维卷积网络提取局部特征模式；
- `LSTMFaultClassifier`：将标准化后的时频特征切分为多步特征序列送入双向/单向 LSTM 网络建模跨特征依赖。

设计特点：
1. CPU / GPU 自动适配，可配置 `epochs`、`batch_size`、`lr` 等超参，保证 CPU 模拟数据秒级训练；
2. 完整实现 `fit` / `predict` / `predict_proba` / `classify_one` / `evaluate` / `save` / `load`，
   可直接无缝接入 `ModelBundle`、`EquipmentToolkit` 与 Streamlit 工作台进行实时模型切换。
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from equipment_agent.features.extractor import FEATURE_COLUMNS
from equipment_agent.models.base import BaseFaultClassifier
from equipment_agent.models.classifier import (
    FaultClassificationResult,
    RandomForestFaultClassifier,
)


FAULT_CLASSIFIER_FILES: dict[str, str] = {
    "rf": "rf_fault_classifier.joblib",
    "cnn1d": "cnn1d_fault_classifier.pt",
    "lstm": "lstm_fault_classifier.pt",
}

FAULT_CLASSIFIER_LABELS: dict[str, str] = {
    "rf": "随机森林 (RF)",
    "cnn1d": "1D-CNN (PyTorch)",
    "lstm": "LSTM (PyTorch)",
}


def _set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _CNN1DNet(nn.Module):
    """轻量级一维卷积网络，面向 54 维（或任意维度）时频特征向量分类。"""

    def __init__(self, input_dim: int, num_classes: int, hidden_channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.features = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2) if input_dim >= 4 else nn.Identity(),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> (B, 1, D)
        feat = self.features(x.unsqueeze(1))
        return self.head(feat)


class _LSTMNet(nn.Module):
    """基于 LSTM 的多步特征序列分类网络。

    将标准化后的 D 维时频特征按 `seq_len` 等分（不足自动零填充）为 `(B, seq_len, step_dim)`，
    例如 54 维 -> 6 步 × 9 维（涵盖水平时域、水平频域、水平频带、垂直时域、垂直频域、垂直频带），
    使用 LSTM 捕获不同物理子空间与双通道之间的序列关联。
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = max(1, min(seq_len, input_dim))
        self.step_dim = int(math.ceil(input_dim / self.seq_len))
        self.padded_dim = self.seq_len * self.step_dim

        self.lstm = nn.LSTM(
            input_size=self.step_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        if x.size(1) < self.padded_dim:
            pad = torch.zeros(bsz, self.padded_dim - x.size(1), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        seq = x.view(bsz, self.seq_len, self.step_dim)
        out, _ = self.lstm(seq)
        # 融合末步隐藏态与全序列均值池化，增强泛化稳定性
        pooled = 0.5 * out[:, -1, :] + 0.5 * out.mean(dim=1)
        return self.head(pooled)


class _BaseTorchFaultClassifier(BaseFaultClassifier):
    """PyTorch 轴承故障分类器共享基类。"""

    MODEL_TYPE: str = "torch_base"
    DISPLAY_NAME: str = "TorchFaultClassifier"

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        epochs: int = 35,
        batch_size: int = 32,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.random_state = int(random_state)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.scaler = StandardScaler()
        self.model: nn.Module | None = None
        self.classes_: list[str] = []
        self.history: list[dict[str, float]] = []
        self.train_time_seconds: float = 0.0
        self._fitted: bool = False

    def _build_net(self, input_dim: int, num_classes: int) -> nn.Module:
        raise NotImplementedError

    def _extra_config(self) -> dict[str, Any]:
        return {}

    def _prepare_matrix(self, x: pd.DataFrame | np.ndarray | dict[str, float]) -> np.ndarray:
        if isinstance(x, dict):
            return np.array([[float(x[c]) for c in self.feature_columns]], dtype=np.float64)
        if isinstance(x, pd.DataFrame):
            return x[self.feature_columns].to_numpy(dtype=np.float64)
        arr = np.asarray(x, dtype=np.float64)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    def fit(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "_BaseTorchFaultClassifier":
        _set_torch_seed(self.random_state)
        t0 = time.perf_counter()

        mat = self._prepare_matrix(x)
        labels_str = np.asarray(y, dtype=str)
        self.classes_ = sorted(set(labels_str.tolist()))
        cls_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([cls_to_idx[c] for c in labels_str], dtype=np.int64)

        scaled = self.scaler.fit_transform(mat).astype(np.float32)

        # 类别均衡权重，避免样本不均衡导致小类召回偏低
        counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(np.float32)
        weights = len(y_idx) / np.maximum(counts * len(self.classes_), 1.0)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(torch.from_numpy(scaled), torch.from_numpy(y_idx))
        effective_bs = max(2, min(self.batch_size, len(dataset)))
        # BatchNorm1d 在最后一个 batch 仅 1 个样本时会报错，故样本数大于 batch_size 时 drop_last
        drop_last = (len(dataset) > effective_bs) and (len(dataset) % effective_bs == 1)
        loader = DataLoader(dataset, batch_size=effective_bs, shuffle=True, drop_last=drop_last)

        self.model = self._build_net(input_dim=scaled.shape[1], num_classes=len(self.classes_))
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.epochs)
        )

        self.history = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for bx, by in loader:
                bx = bx.to(self.device)
                by = by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * bx.size(0)
                correct += int((logits.argmax(dim=1) == by).sum().item())
                total += int(by.size(0))

            scheduler.step()
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(total_loss / max(1, total), 4),
                    "train_acc": round(correct / max(1, total), 4),
                }
            )

        self.model.eval()
        self.train_time_seconds = round(time.perf_counter() - t0, 4)
        self._fitted = True
        return self

    @torch.no_grad()
    def _predict_proba_array(self, x: pd.DataFrame | np.ndarray | dict[str, float]) -> np.ndarray:
        if not self._fitted or self.model is None:
            raise RuntimeError(f"{self.DISPLAY_NAME} 尚未训练，请先调用 fit() 或 load()。")
        mat = self._prepare_matrix(x)
        scaled = self.scaler.transform(mat).astype(np.float32)
        tensor = torch.from_numpy(scaled).to(self.device)
        self.model.eval()
        logits = self.model(tensor)
        return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, x: pd.DataFrame | np.ndarray | dict[str, float]) -> np.ndarray:
        probs = self._predict_proba_array(x)
        indices = np.argmax(probs, axis=1)
        return np.array([self.classes_[i] for i in indices], dtype=str)

    def predict_proba(
        self, x: pd.DataFrame | np.ndarray | dict[str, float]
    ) -> list[dict[str, float]]:
        probs = self._predict_proba_array(x)
        results: list[dict[str, float]] = []
        for row in probs:
            results.append(
                {cls_name: round(float(p), 4) for cls_name, p in zip(self.classes_, row)}
            )
        return results

    def classify_one(
        self, features: dict[str, float] | pd.Series | np.ndarray
    ) -> FaultClassificationResult:
        """对单个采样文件执行故障分类，返回最优故障类型与概率分布。"""
        if isinstance(features, pd.Series):
            features = features.to_dict()
        probs = self.predict_proba(features)[0]
        sorted_items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        best_fault, best_conf = sorted_items[0]
        return FaultClassificationResult(
            fault_type=best_fault,
            confidence=round(float(best_conf), 4),
            probabilities=probs,
            model_name=self.DISPLAY_NAME,
        )

    def evaluate(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> dict[str, Any]:
        y_true = np.asarray(y, dtype=str)
        t0 = time.perf_counter()
        y_pred = self.predict(x)
        infer_sec = time.perf_counter() - t0
        n_samples = max(1, len(y_true))
        labels = sorted(set(y_true) | set(y_pred))
        return {
            "model_name": self.DISPLAY_NAME,
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "report": classification_report(
                y_true, y_pred, labels=labels, output_dict=True, zero_division=0
            ),
            "train_time_seconds": round(float(self.train_time_seconds), 4),
            "infer_time_seconds": round(float(infer_sec), 5),
            "infer_time_ms_per_sample": round((infer_sec / n_samples) * 1000.0, 4),
        }

    def save(self, path: str | Path) -> Path:
        if not self._fitted or self.model is None:
            raise RuntimeError(f"{self.DISPLAY_NAME} 尚未训练，无法保存。")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.MODEL_TYPE,
            "feature_columns": self.feature_columns,
            "classes_": self.classes_,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "random_state": self.random_state,
            "extra_config": self._extra_config(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "scaler_var": self.scaler.var_.tolist(),
            "scaler_n_samples": int(self.scaler.n_samples_seen_),
            "state_dict": self.model.cpu().state_dict(),
            "history": self.history,
            "train_time_seconds": self.train_time_seconds,
        }
        torch.save(payload, target)
        self.model.to(self.device)
        return target

    @classmethod
    def _restore_scaler(cls, payload: dict[str, Any]) -> StandardScaler:
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(payload["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=np.float64)
        scaler.var_ = np.asarray(payload["scaler_var"], dtype=np.float64)
        scaler.n_samples_seen_ = int(payload["scaler_n_samples"])
        scaler.n_features_in_ = len(payload["feature_columns"])
        return scaler


class CNN1DFaultClassifier(_BaseTorchFaultClassifier):
    """基于 PyTorch 一维卷积神经网络（1D-CNN）的轴承故障分类器。"""

    MODEL_TYPE = "cnn1d"
    DISPLAY_NAME = "CNN1DFaultClassifier"

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        hidden_channels: int = 32,
        epochs: int = 35,
        batch_size: int = 32,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__(
            feature_columns=feature_columns,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            random_state=random_state,
            device=device,
        )
        self.hidden_channels = int(hidden_channels)

    def _build_net(self, input_dim: int, num_classes: int) -> nn.Module:
        return _CNN1DNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_channels=self.hidden_channels,
            dropout=self.dropout,
        )

    def _extra_config(self) -> dict[str, Any]:
        return {"hidden_channels": self.hidden_channels}

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "CNN1DFaultClassifier":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        extra = payload.get("extra_config", {})
        obj = cls(
            feature_columns=payload["feature_columns"],
            hidden_channels=int(extra.get("hidden_channels", 32)),
            epochs=int(payload.get("epochs", 35)),
            batch_size=int(payload.get("batch_size", 32)),
            lr=float(payload.get("lr", 2e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            dropout=float(payload.get("dropout", 0.2)),
            random_state=int(payload.get("random_state", 42)),
            device=device,
        )
        obj.classes_ = list(payload["classes_"])
        obj.scaler = cls._restore_scaler(payload)
        obj.model = obj._build_net(input_dim=len(obj.feature_columns), num_classes=len(obj.classes_))
        obj.model.load_state_dict(payload["state_dict"])
        obj.model.to(obj.device)
        obj.model.eval()
        obj.history = list(payload.get("history", []))
        obj.train_time_seconds = float(payload.get("train_time_seconds", 0.0))
        obj._fitted = True
        return obj


class LSTMFaultClassifier(_BaseTorchFaultClassifier):
    """基于 PyTorch LSTM 的轴承故障分类器。"""

    MODEL_TYPE = "lstm"
    DISPLAY_NAME = "LSTMFaultClassifier"

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        seq_len: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 40,
        batch_size: int = 32,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__(
            feature_columns=feature_columns,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            random_state=random_state,
            device=device,
        )
        self.seq_len = int(seq_len)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

    def _build_net(self, input_dim: int, num_classes: int) -> nn.Module:
        return _LSTMNet(
            input_dim=input_dim,
            num_classes=num_classes,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def _extra_config(self) -> dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "LSTMFaultClassifier":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        extra = payload.get("extra_config", {})
        obj = cls(
            feature_columns=payload["feature_columns"],
            seq_len=int(extra.get("seq_len", 6)),
            hidden_size=int(extra.get("hidden_size", 64)),
            num_layers=int(extra.get("num_layers", 2)),
            epochs=int(payload.get("epochs", 40)),
            batch_size=int(payload.get("batch_size", 32)),
            lr=float(payload.get("lr", 2e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            dropout=float(payload.get("dropout", 0.2)),
            random_state=int(payload.get("random_state", 42)),
            device=device,
        )
        obj.classes_ = list(payload["classes_"])
        obj.scaler = cls._restore_scaler(payload)
        obj.model = obj._build_net(input_dim=len(obj.feature_columns), num_classes=len(obj.classes_))
        obj.model.load_state_dict(payload["state_dict"])
        obj.model.to(obj.device)
        obj.model.eval()
        obj.history = list(payload.get("history", []))
        obj.train_time_seconds = float(payload.get("train_time_seconds", 0.0))
        obj._fitted = True
        return obj


def load_fault_classifier(
    model_key: str,
    artifact_dir: str | Path,
    device: str | None = None,
) -> BaseFaultClassifier:
    """按模型标识（`rf` / `cnn1d` / `lstm`）从目录加载对应故障分类器。"""
    key = model_key.strip().lower()
    if key not in FAULT_CLASSIFIER_FILES:
        raise ValueError(f"不支持的故障分类模型类型：{model_key}，可选：{list(FAULT_CLASSIFIER_FILES.keys())}")
    target = Path(artifact_dir) / FAULT_CLASSIFIER_FILES[key]
    if not target.exists():
        raise FileNotFoundError(f"未找到模型文件：{target}")
    if key == "rf":
        return RandomForestFaultClassifier.load(target)
    if key == "cnn1d":
        return CNN1DFaultClassifier.load(target, device=device)
    return LSTMFaultClassifier.load(target, device=device)
