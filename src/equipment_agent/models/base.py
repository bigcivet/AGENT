"""诊断模型抽象基类定义。

统一异常检测、故障分类、剩余寿命（RUL）预测三类模型的接口契约，
当前基线采用 Isolation Forest + 随机森林（Random Forest），
后续切换为 PyTorch CNN / LSTM 时只需继承以下基类并实现相同方法，
智能体编排层与 Web 前端无需改动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseAnomalyDetector(ABC):
    """无监督异常检测模型基类。"""

    @abstractmethod
    def fit(self, x: pd.DataFrame | np.ndarray) -> "BaseAnomalyDetector":
        """基于设备健康阶段特征拟合异常检测模型。"""

    @abstractmethod
    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """返回布尔数组，True 表示检测到异常，False 表示正常。"""

    @abstractmethod
    def score(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """返回异常程度分数（越大表示越异常）。"""

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """持久化模型到磁盘。"""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseAnomalyDetector":
        """从磁盘加载模型。"""


class BaseFaultClassifier(ABC):
    """轴承故障分类模型基类（便于后续从随机森林切换为 CNN / 1D-ResNet）。"""

    @abstractmethod
    def fit(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "BaseFaultClassifier":
        """训练故障分类器。"""

    @abstractmethod
    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """预测故障类别标签。"""

    @abstractmethod
    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> dict[str, float] | list[dict[str, float]]:
        """返回各类别预测概率分布。"""

    @abstractmethod
    def evaluate(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> dict[str, Any]:
        """在测试集上评估分类性能（准确率、F1、混淆矩阵等）。"""

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """保存模型。"""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseFaultClassifier":
        """加载模型。"""


class BaseRULPredictor(ABC):
    """剩余寿命（RUL）预测模型基类（便于后续从随机森林切换为 LSTM / Transformer）。"""

    @abstractmethod
    def fit(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "BaseRULPredictor":
        """训练剩余寿命回归模型。"""

    @abstractmethod
    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """预测剩余寿命（相对比例或绝对分钟数）。"""

    @abstractmethod
    def evaluate(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> dict[str, float]:
        """在测试集上评估回归性能（MAE、RMSE、R²）。"""

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """保存模型。"""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseRULPredictor":
        """加载模型。"""
