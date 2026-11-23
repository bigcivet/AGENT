"""项目全局配置模块。

支持从环境变量和项目根目录 `.env` 文件读取配置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 兜底处理，未装 python-dotenv 时仍可运行
    load_dotenv = None


# 项目根目录（src/equipment_agent/config.py -> 上溯三层）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)


@dataclass(frozen=True)
class AppConfig:
    """应用运行配置。"""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data" / "xjtu_sy"
    artifact_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    sampling_rate: int = 25600
    expected_samples_per_file: int = 32768

    # OpenAI 兼容大模型接口配置（可对接 DeepSeek / 通义千问 / OpenAI / vLLM 等）
    openai_api_key: str = ""
    openai_base_url: str = "https://api.deepseek.com/v1"
    openai_model: str = "deepseek-chat"
    llm_timeout_seconds: float = 30.0

    @property
    def has_llm_key(self) -> bool:
        """是否已配置有效的大模型 API Key。"""
        key = self.openai_api_key.strip()
        return bool(key) and not key.startswith("sk-your-")


def _resolve_path(raw: str | None, default: Path) -> Path:
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config() -> AppConfig:
    """从环境变量加载最新配置。"""
    return AppConfig(
        project_root=PROJECT_ROOT,
        data_dir=_resolve_path(os.getenv("XJTU_DATA_DIR"), PROJECT_ROOT / "data" / "xjtu_sy"),
        artifact_dir=_resolve_path(
            os.getenv("MODEL_ARTIFACT_DIR"), PROJECT_ROOT / "artifacts" / "models"
        ),
        sampling_rate=int(os.getenv("SAMPLING_RATE", "25600")),
        expected_samples_per_file=int(os.getenv("SAMPLES_PER_FILE", "32768")),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "deepseek-chat").strip(),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0")),
    )
