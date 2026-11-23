"""视觉质检智能体（基于 NEU-DET 金属表面缺陷数据）。

模块组成：
- vision: NEU-DET 数据加载、ResNet18 迁移学习分类与可选 YOLO 检测适配器
- memory: 基于 SQLite 的产线质量“数字孪生”记忆库
- decision: 连续缺陷主动预警引擎与轻量级工艺文档 RAG 检索
- agent: 工具集（统计/受限只读 SQL/RAG）与自然语言质检推理编排器
"""

from vision_agent.config import (
    DEFECT_CLASSES,
    DEFECT_CN_MAP,
    VisionConfig,
    load_vision_config,
)

__all__ = [
    "DEFECT_CLASSES",
    "DEFECT_CN_MAP",
    "VisionConfig",
    "load_vision_config",
]
