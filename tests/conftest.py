"""pytest 公共配置与共享夹具。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from equipment_agent.data.loader import XJTUSYDataLoader
from equipment_agent.models.trainer import ModelBundle, TrainingPipeline
from generate_mock_data import generate_mock_xjtu_dataset


@pytest.fixture(scope="session")
def mock_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """生成轻量级模拟 XJTU-SY 数据集，供整个测试会话复用。"""
    data_dir = tmp_path_factory.mktemp("mock_xjtu_sy")
    generate_mock_xjtu_dataset(
        output_dir=data_dir,
        files_per_bearing=16,
        samples_per_file=2048,
        sampling_rate=25600,
        seed=123,
    )
    return data_dir


@pytest.fixture(scope="session")
def trained_env(
    mock_dataset_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[XJTUSYDataLoader, ModelBundle, Path]:
    """在模拟数据集上完成一次完整训练并返回 `(loader, bundle, artifact_dir)`。"""
    artifact_dir = tmp_path_factory.mktemp("mock_models")
    loader = XJTUSYDataLoader(data_dir=mock_dataset_dir, sampling_rate=25600)
    pipeline = TrainingPipeline(loader=loader, artifact_dir=artifact_dir, random_state=42)
    bundle = pipeline.run()
    return loader, bundle, artifact_dir
