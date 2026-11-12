"""
IS403.R14 - Nhóm 8 - Phân tích và dự báo xe điện Washington
Thành viên TV2: Nguyễn Nữ Trà Giang
Package common: Chứa module tạo đặc trưng (features) và khung backtest (backtest) dùng chung cho Chương 5.
"""

from .features import (
    compute_log_growth,
    create_features,
    build_supervised_data,
    get_feature_names,
    compute_step_features_from_buffer,
    FEATURE_METADATA,
)

from .backtest import (
    FOLDS_DEFINITION,
    fit,
    forecast,
    fit_model,
    forecast_recursive,
    run_backtest,
    inverse_transform_growth,
)

__all__ = [
    "compute_log_growth",
    "create_features",
    "build_supervised_data",
    "get_feature_names",
    "compute_step_features_from_buffer",
    "FEATURE_METADATA",
    "FOLDS_DEFINITION",
    "fit",
    "forecast",
    "fit_model",
    "forecast_recursive",
    "run_backtest",
    "inverse_transform_growth",
]
