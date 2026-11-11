"""
Module: common/backtest.py
Tác giả: Nguyễn Nữ Trà Giang (TV2) - Nhóm 8 - IS403.R14
Mô tả: Khung thực nghiệm backtest theo chuỗi thời gian (Time-series Backtesting)
      và động cơ dự báo đệ quy 12 bước (Recursive multi-step forecasting)
      dùng chung cho 4 mô hình: Ridge, Random Forest, SVR, XGBoost.
      Đảm bảo 100% không rò rỉ dữ liệu tương lai (Zero Data Leakage).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Nhập các hàm trích xuất đặc trưng từ module features
from .features import (
    compute_log_growth,
    create_features,
    build_supervised_data,
    compute_step_features_from_buffer,
    get_feature_names,
)

# Thử import xgboost nếu môi trường đã cài đặt
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


FOLDS_DEFINITION: List[Dict[str, Any]] = [
    {
        "fold": 1,
        "origin": "2020-11",
        "train_end": "2020-11-30",
        "val_start": "2020-12-31",
        "val_end": "2021-11-30",
        "n_train_months": 47,
        "n_supervised_rows": 40,
        "val_months": 12,
    },
    {
        "fold": 2,
        "origin": "2021-05",
        "train_end": "2021-05-31",
        "val_start": "2021-06-30",
        "val_end": "2022-05-31",
        "n_train_months": 53,
        "n_supervised_rows": 46,
        "val_months": 12,
    },
    {
        "fold": 3,
        "origin": "2021-11",
        "train_end": "2021-11-30",
        "val_start": "2021-12-31",
        "val_end": "2022-11-30",
        "n_train_months": 59,
        "n_supervised_rows": 52,
        "val_months": 12,
    },
]


def inverse_transform_growth(r_preds: Union[List[float], np.ndarray], last_y: float) -> np.ndarray:
    """
    Chuyển đổi chuỗi tăng trưởng log dự báo r_hat về chuỗi số lượng xe y_hat:
        y_{T+h} = y_T * exp( sum_{j=1}^h r_{T+j} )
    Lưu ý: Không kẹp tăng trưởng âm thành 0 để tránh méo mó phân phối.
    """
    r_arr = np.array(r_preds, dtype=float)
    cumsum_r = np.cumsum(r_arr)
    y_preds = float(last_y) * np.exp(cumsum_r)
    return y_preds


def _instantiate_estimator(model_type: str, config: Dict[str, Any]) -> Any:
    """Khởi tạo estimator chuẩn hóa theo cấu hình được cung cấp."""
    m_type = model_type.lower().strip()
    
    if "estimator" in config and config["estimator"] is not None:
        return config["estimator"]
        
    if m_type == "ridge":
        alpha = config.get("alpha", 1.0)
        random_state = config.get("random_state", 42)
        return Ridge(alpha=alpha, random_state=random_state)
        
    elif m_type in ["rf", "random_forest", "randomforest"]:
        n_estimators = config.get("n_estimators", 300)
        max_depth = config.get("max_depth", 2)
        min_samples_leaf = config.get("min_samples_leaf", 3)
        max_features = config.get("max_features", 0.7)
        random_state = config.get("random_state", 42)
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=1,
        )
        
    elif m_type == "svr":
        kernel = config.get("kernel", "rbf")
        C = config.get("C", 1.0)
        epsilon = config.get("epsilon", 0.1)
        gamma = config.get("gamma", "scale")
        return SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        
    elif m_type in ["xgb", "xgboost"]:
        if not HAS_XGBOOST:
            raise ImportError("Thư viện xgboost chưa được cài đặt trong môi trường!")
        max_depth = config.get("max_depth", 1)
        n_estimators = config.get("n_estimators", 50)
        learning_rate = config.get("learning_rate", 0.03)
        min_child_weight = config.get("min_child_weight", 3)
        subsample = config.get("subsample", 0.8)
        colsample_bytree = config.get("colsample_bytree", 1.0)
        reg_lambda = config.get("reg_lambda", 1.0)
        reg_alpha = config.get("reg_alpha", 0.0)
        random_state = config.get("random_state", 42)
        return XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Loại mô hình không được hỗ trợ: {model_type}")


def fit_model(
    train_history: pd.Series,
    config: Optional[Dict[str, Any]] = None,
    model_type: str = "ridge",
    needs_scaling: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Huấn luyện mô hình một bước (one-step ahead) trên dữ liệu lịch sử train_history.
    
    Quy trình chặt chẽ:
    1. Trích xuất r_t và 5 đặc trưng từ train_history.
    2. Loại bỏ các giá trị NaN đầu (cần 7 tháng warm-up).
    3. Fit StandardScaler cho biến mục tiêu r trên tập train của fold này:
           z_t = (r_t - mu_r) / std_r
    4. Nếu mô hình cần chuẩn hóa X (Ridge, SVR): Fit StandardScaler cho X_train.
       Nếu mô hình không cần (RF, XGBoost): Giữ nguyên thang đo gốc của X.
    5. Fit mô hình trên (X_train_fit, z_train).
    6. Đóng gói toàn bộ artifact cần thiết cho khâu dự báo đệ quy.
    """
    if config is None:
        config = {}
        
    m_type = model_type.lower().strip()
    if needs_scaling is None:
        # Thiết kế chung: Ridge/SVR cần chuẩn hóa X, RF/XGBoost không cần
        needs_scaling = m_type in ["ridge", "svr"]
        
    # Đảm bảo train_history là series số thực có index thời gian
    y_train = pd.Series(train_history.values, index=pd.to_datetime(train_history.index), dtype=float)
    y_train = y_train.sort_index()
    n_train_months = len(y_train)
    
    # Tạo chuỗi tăng trưởng r và dữ liệu học có giám sát
    r_full = compute_log_growth(y_train)
    df_sup = build_supervised_data(y_train, dropna=True)
    n_supervised_rows = len(df_sup)
    
    feature_cols = get_feature_names()
    X_train = df_sup[feature_cols].copy()
    r_target = df_sup["r"].values
    
    # 1. Fit scaler cục bộ cho nhãn r
    mu_r = float(np.mean(r_target))
    std_r = float(np.std(r_target, ddof=0))
    if std_r == 0:
        std_r = 1e-8
    z_target = (r_target - mu_r) / std_r
    
    # 2. Fit scaler cho đặc trưng X nếu cần
    if needs_scaling:
        scaler_X = StandardScaler()
        X_train_fit = scaler_X.fit_transform(X_train)
    else:
        scaler_X = None
        X_train_fit = X_train.values
        
    # 3. Khởi tạo và fit mô hình
    estimator = _instantiate_estimator(m_type, config)
    estimator.fit(X_train_fit, z_target)
    
    # 4. Lưu lại 6 giá trị r gần nhất và giá trị y cuối cùng để làm buffer dự báo
    r_clean = r_full.dropna()
    last_r_history = list(r_clean.iloc[-6:].values)
    last_y = float(y_train.iloc[-1])
    train_end_date = str(y_train.index[-1].strftime("%Y-%m-%d"))
    
    artifact: Dict[str, Any] = {
        "model": estimator,
        "model_type": m_type,
        "scaler_X": scaler_X,
        "mu_r": mu_r,
        "std_r": std_r,
        "last_y": last_y,
        "last_r_history": last_r_history,
        "train_end_date": train_end_date,
        "n_train_months": n_train_months,
        "n_supervised_rows": n_supervised_rows,
        "feature_names": feature_cols,
        "config": config,
        "needs_scaling": needs_scaling,
    }
    
    return artifact


def forecast_recursive(
    artifact: Dict[str, Any],
    train_history: Optional[pd.Series] = None,
    horizon: int = 12,
) -> Dict[str, Any]:
    """
    Thực hiện dự báo đệ quy đa bước (Recursive multi-step forecast) cho horizon bước (mặc định 12).
    
    Cam kết không rò rỉ:
    - KHÔNG nhận bất kỳ thông tin thực tế nào của tương lai.
    - Tại mỗi bước h (1..12), đặc trưng được tính hoàn toàn từ buffer (chứa lịch sử và các dự báo trước).
    - Dự báo ra z_hat -> đổi về r_hat -> bổ sung vào buffer -> tính lũy kế ra y_hat.
    """
    estimator = artifact["model"]
    scaler_X = artifact["scaler_X"]
    mu_r = artifact["mu_r"]
    std_r = artifact["std_r"]
    last_y = artifact["last_y"]
    buffer_r = list(artifact["last_r_history"])
    
    z_preds: List[float] = []
    r_preds: List[float] = []
    trace_records: List[Dict[str, Any]] = []
    
    for h in range(1, horizon + 1):
        # 1. Trích xuất đặc trưng bước h từ buffer
        step_features = compute_step_features_from_buffer(buffer_r)
        step_features_df = pd.DataFrame([step_features], columns=artifact["feature_names"])
        
        # 2. Chuẩn hóa đặc trưng nếu mô hình yêu cầu
        if scaler_X is not None:
            features_for_model = scaler_X.transform(step_features_df)
        else:
            features_for_model = step_features_df.values
            
        # 3. Dự đoán z_hat
        z_hat = float(estimator.predict(features_for_model)[0])
        z_preds.append(z_hat)
        
        # 4. Đổi ngược z_hat -> r_hat
        r_hat = float(z_hat * std_r + mu_r)
        r_preds.append(r_hat)
        
        # 5. Cập nhật buffer cho bước kế tiếp
        buffer_r.append(r_hat)
        
        # 6. Ghi trace phục vụ kiểm tra
        trace_record = {
            "step": h,
            "r_lag1": step_features[0],
            "r_lag2": step_features[1],
            "r_lag3": step_features[2],
            "r_roll3": step_features[3],
            "r_roll6": step_features[4],
            "z_pred": z_hat,
            "r_pred": r_hat,
        }
        trace_records.append(trace_record)
        
    # 7. Chuyển đổi toàn bộ r_preds sang y_preds
    y_preds = inverse_transform_growth(r_preds, last_y)
    
    # Gắn thêm y_pred vào trace_df
    for i, rec in enumerate(trace_records):
        rec["y_pred"] = float(y_preds[i])
        
    trace_df = pd.DataFrame(trace_records)
    
    return {
        "y_pred": np.array(y_preds, dtype=float),
        "r_pred": np.array(r_preds, dtype=float),
        "z_pred": np.array(z_preds, dtype=float),
        "trace_df": trace_df,
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính toán 3 chỉ số đánh giá cốt lõi: RMSE, MAE và sMAPE."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    
    denom = np.abs(yt) + np.abs(yp)
    # Xử lý trường hợp mẫu số bằng 0
    zero_mask = denom == 0
    ratio = np.zeros_like(denom)
    ratio[~zero_mask] = 2.0 * np.abs(yt[~zero_mask] - yp[~zero_mask]) / denom[~zero_mask]
    smape = float(100.0 * np.mean(ratio))
    
    bias = float(np.mean(yt - yp))
    
    return {
        "rmse": rmse,
        "mae": mae,
        "smape": smape,
        "bias": bias,
    }


def run_backtest(
    y_series: pd.Series,
    config: Optional[Dict[str, Any]] = None,
    model_type: str = "ridge",
    folds_def: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Chạy thực nghiệm backtest 3-fold expanding window cho một cấu hình mô hình cụ thể.
    Trả về:
        - cv_results_df: Kết quả chi tiết theo từng fold (tương thích cv_<model>.csv)
        - summary_metrics: Trung bình các chỉ số qua 3 folds
    """
    if config is None:
        config = {}
    if folds_def is None:
        folds_def = FOLDS_DEFINITION
        
    y_s = pd.Series(y_series.values, index=pd.to_datetime(y_series.index), dtype=float)
    y_s = y_s.sort_index()
    
    config_id = config.get("config_id", f"{model_type}_default")
    fold_rows = []
    all_fold_preds: List[np.ndarray] = []
    
    for f in folds_def:
        fold_num = f["fold"]
        train_end = pd.to_datetime(f["train_end"])
        val_start = pd.to_datetime(f["val_start"])
        val_end = pd.to_datetime(f["val_end"])
        
        # Slices không rò rỉ
        train_mask = y_s.index <= train_end
        val_mask = (y_s.index >= val_start) & (y_s.index <= val_end)
        
        train_data = y_s[train_mask]
        val_data = y_s[val_mask]
        
        # 1. Fit trên train của fold này
        artifact = fit_model(train_data, config=config, model_type=model_type)
        
        # 2. Forecast đệ quy 12 bước
        fc_res = forecast_recursive(artifact, horizon=len(val_data))
        y_pred = fc_res["y_pred"]
        all_fold_preds.append(y_pred)
        
        # 3. Tính điểm trên thang số xe
        metrics = calculate_metrics(val_data.values, y_pred)
        
        row = {
            "config_id": config_id,
            "fold": fold_num,
            "origin": f["origin"],
            "n_train_months": artifact["n_train_months"],
            "n_supervised_rows": artifact["n_supervised_rows"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "smape": metrics["smape"],
            "bias": metrics["bias"],
            "status": "SUCCESS",
            "warnings": "None",
        }
        fold_rows.append(row)
        
    cv_df = pd.DataFrame(fold_rows)
    
    summary = {
        "mean_cv_rmse": float(cv_df["rmse"].mean()),
        "mean_cv_mae": float(cv_df["mae"].mean()),
        "mean_cv_smape": float(cv_df["smape"].mean()),
        "mean_cv_bias": float(cv_df["bias"].mean()),
    }
    
    return cv_df, summary
