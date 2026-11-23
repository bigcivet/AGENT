"""视觉质检智能体全局配置模块。

复用 `equipment_agent.config` 中的项目根目录、`.env` 加载与 OpenAI 兼容大模型配置，
避免重复维护环境变量。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from equipment_agent.config import PROJECT_ROOT, AppConfig, load_config

# NEU-DET 标准 6 类金属表面缺陷（按字母顺序，与 PyTorch ImageFolder 保持一致）
DEFECT_CLASSES: tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)

# 英文缺陷类别 -> 中文标准名称
DEFECT_CN_MAP: dict[str, str] = {
    "crazing": "裂纹",
    "inclusion": "夹杂",
    "patches": "斑块",
    "pitted_surface": "麻点",
    "rolled-in_scale": "压入氧化皮",
    "scratches": "划痕",
}

# 自然语言别名映射（用于中文问答解析与 VOC 标签标准化）
DEFECT_ALIAS_MAP: dict[str, str] = {
    "crazing": "crazing",
    "裂纹": "crazing",
    "龟裂": "crazing",
    "微裂纹": "crazing",
    "inclusion": "inclusion",
    "夹杂": "inclusion",
    "夹渣": "inclusion",
    "patches": "patches",
    "斑块": "patches",
    "色斑": "patches",
    "暗斑": "patches",
    "pitted_surface": "pitted_surface",
    "pitted": "pitted_surface",
    "麻点": "pitted_surface",
    "点蚀": "pitted_surface",
    "凹坑": "pitted_surface",
    "rolled-in_scale": "rolled-in_scale",
    "rolled_in_scale": "rolled-in_scale",
    "scale": "rolled-in_scale",
    "压入氧化皮": "rolled-in_scale",
    "氧化皮": "rolled-in_scale",
    "氧化铁皮": "rolled-in_scale",
    "scratches": "scratches",
    "scratch": "scratches",
    "划痕": "scratches",
    "划伤": "scratches",
    "擦伤": "scratches",
}

# 默认严重等级映射（工艺先验）
DEFECT_SEVERITY_MAP: dict[str, str] = {
    "crazing": "高",
    "inclusion": "高",
    "scratches": "高",
    "rolled-in_scale": "中",
    "pitted_surface": "中",
    "patches": "低",
}


def normalize_defect_name(raw: str) -> str:
    """将中英文缺陷别名规范化为 NEU-DET 标准英文类名。"""
    key = raw.strip().lower().replace(" ", "_")
    if key in DEFECT_ALIAS_MAP:
        return DEFECT_ALIAS_MAP[key]
    for alias, canonical in DEFECT_ALIAS_MAP.items():
        if alias in key:
            return canonical
    return raw.strip()


def defect_to_cn(defect_type: str) -> str:
    """获取缺陷的中文显示名称。"""
    canonical = normalize_defect_name(defect_type)
    return DEFECT_CN_MAP.get(canonical, defect_type)


def _resolve_path(raw: str | None, default: Path) -> Path:
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class VisionConfig:
    """视觉质检智能体运行配置。"""

    project_root: Path = PROJECT_ROOT
    neu_det_dir: Path = PROJECT_ROOT / "data" / "neu_det"
    vision_artifact_dir: Path = PROJECT_ROOT / "artifacts" / "vision_models"
    db_path: Path = PROJECT_ROOT / "data" / "vision_memory.db"
    process_docs_dir: Path = PROJECT_ROOT / "data" / "process_docs"
    alert_consecutive_threshold: int = 3
    image_size: int = 224
    app_config: AppConfig = field(default_factory=load_config)

    @property
    def has_llm_key(self) -> bool:
        return self.app_config.has_llm_key

    @property
    def openai_api_key(self) -> str:
        return self.app_config.openai_api_key

    @property
    def openai_base_url(self) -> str:
        return self.app_config.openai_base_url

    @property
    def openai_model(self) -> str:
        return self.app_config.openai_model

    @property
    def llm_timeout_seconds(self) -> float:
        return self.app_config.llm_timeout_seconds


def load_vision_config() -> VisionConfig:
    """从环境变量加载视觉质检配置（并复用统一大模型配置）。"""
    base_cfg = load_config()
    return VisionConfig(
        project_root=PROJECT_ROOT,
        neu_det_dir=_resolve_path(os.getenv("NEU_DET_DATA_DIR"), PROJECT_ROOT / "data" / "neu_det"),
        vision_artifact_dir=_resolve_path(
            os.getenv("VISION_ARTIFACT_DIR"), PROJECT_ROOT / "artifacts" / "vision_models"
        ),
        db_path=_resolve_path(os.getenv("VISION_DB_PATH"), PROJECT_ROOT / "data" / "vision_memory.db"),
        process_docs_dir=_resolve_path(
            os.getenv("PROCESS_DOCS_DIR"), PROJECT_ROOT / "data" / "process_docs"
        ),
        alert_consecutive_threshold=int(os.getenv("VISION_ALERT_THRESHOLD", "3")),
        image_size=int(os.getenv("VISION_IMAGE_SIZE", "224")),
        app_config=base_cfg,
    )
