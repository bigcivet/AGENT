"""可复用的 OpenAI 兼容大模型客户端封装。

直接读取 `equipment_agent.config.AppConfig`，供设备运维与视觉质检智能体共用。
"""

from __future__ import annotations

from typing import Any

from equipment_agent.config import AppConfig, load_config


def create_openai_client(config: AppConfig | None = None) -> Any:
    """根据统一配置创建 OpenAI 兼容客户端（支持 DeepSeek、通义千问、OpenAI 等）。"""
    from openai import OpenAI

    cfg = config or load_config()
    return OpenAI(
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
        timeout=cfg.llm_timeout_seconds,
    )
