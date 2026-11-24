"""时域与频域振动特征提取与特征分析子包。"""

from .analysis import (
    FeatureCorrelationReport,
    PCADimensionalityReducer,
    compute_feature_correlation,
)
from .extractor import (
    FEATURE_COLUMNS,
    build_feature_dataset,
    compute_fft_spectrum,
    extract_channel_features,
    extract_dual_channel_features,
    extract_sample_features,
)

__all__ = [
    "FEATURE_COLUMNS",
    "FeatureCorrelationReport",
    "PCADimensionalityReducer",
    "build_feature_dataset",
    "compute_feature_correlation",
    "compute_fft_spectrum",
    "extract_channel_features",
    "extract_dual_channel_features",
    "extract_sample_features",
]
