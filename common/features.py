"""
Module: common/features.py
Tác giả: Nguyễn Nữ Trà Giang (TV2) - Nhóm 8 - IS403.R14
Mô tả: Feature engineering cho bài toán dự báo quy mô xe điện (Chương 5).
      Chuyển đổi chuỗi thời gian số lượng xe y_t sang chuỗi tăng trưởng log r_t,
      sau đó tạo 5 đặc trưng quá khứ (lag & rolling) tuân thủ nghiêm ngặt nguyên tắc
      không rò rỉ dữ liệu tương lai (shift(1) trước khi rolling).
"""

from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd


FEATURE_NAMES = ["r_lag1", "r_lag2", "r_lag3", "r_roll3", "r_roll6"]

FEATURE_METADATA: List[Dict[str, Any]] = [
    {
        "feature_id": 1,
        "feature_name": "r_lag1",
        "display_name": "Độ trễ bậc 1 (r_{t-1})",
        "formula": "r_{t-1}",
        "shift_steps": 1,
        "window_size": 1,
        "description": "Tốc độ tăng trưởng log ở tháng liền trước (t-1). Nắm bắt động lượng ngắn hạn gần nhất.",
        "scale_requirement": "Có cho Ridge/SVR; Không bắt buộc cho RF/XGBoost",
    },
    {
        "feature_id": 2,
        "feature_name": "r_lag2",
        "display_name": "Độ trễ bậc 2 (r_{t-2})",
        "formula": "r_{t-2}",
        "shift_steps": 2,
        "window_size": 1,
        "description": "Tốc độ tăng trưởng log ở 2 tháng trước (t-2). Phản ánh xu hướng trễ bậc 2.",
        "scale_requirement": "Có cho Ridge/SVR; Không bắt buộc cho RF/XGBoost",
    },
    {
        "feature_id": 3,
        "feature_name": "r_lag3",
        "display_name": "Độ trễ bậc 3 (r_{t-3})",
        "formula": "r_{t-3}",
        "shift_steps": 3,
        "window_size": 1,
        "description": "Tốc độ tăng trưởng log ở 3 tháng trước (t-3). Nắm bắt chu kỳ quý gần nhất.",
        "scale_requirement": "Có cho Ridge/SVR; Không bắt buộc cho RF/XGBoost",
    },
    {
        "feature_id": 4,
        "feature_name": "r_roll3",
        "display_name": "Trung bình trượt 3 tháng (Rolling Mean 3)",
        "formula": "(r_{t-1} + r_{t-2} + r_{t-3}) / 3",
        "shift_steps": 1,
        "window_size": 3,
        "description": "Trung bình tốc độ tăng trưởng của 3 tháng trước đó (shift(1) rồi rolling). Làm mượt nhiễu quý.",
        "scale_requirement": "Có cho Ridge/SVR; Không bắt buộc cho RF/XGBoost",
    },
    {
        "feature_id": 5,
        "feature_name": "r_roll6",
        "display_name": "Trung bình trượt 6 tháng (Rolling Mean 6)",
        "formula": "(r_{t-1} + ... + r_{t-6}) / 6",
        "shift_steps": 1,
        "window_size": 6,
        "description": "Trung bình tốc độ tăng trưởng của 6 tháng trước đó (shift(1) rồi rolling). Phản ánh xu thế trung hạn.",
        "scale_requirement": "Có cho Ridge/SVR; Không bắt buộc cho RF/XGBoost",
    },
]


def compute_log_growth(y: pd.Series) -> pd.Series:
    """
    Tính tốc độ tăng trưởng log r_t:
        r_t = log(y_t) - log(y_{t-1}) = log(y_t / y_{t-1})
    """
    y_float = y.astype(float)
    log_y = np.log(y_float)
    r = log_y.diff()
    r.name = "r"
    return r


def create_features(r: pd.Series) -> pd.DataFrame:
    """
    Tạo 5 đặc trưng từ chuỗi tăng trưởng r_t.
    Lưu ý then chốt: Rolling mean bắt buộc phải shift(1) trước khi tính
    để đảm bảo tại thời điểm t, đặc trưng chỉ chứa thông tin từ t-1 trở về trước.
    """
    r_series = pd.Series(r, copy=True)
    
    # Lag 1, 2, 3
    r_lag1 = r_series.shift(1)
    r_lag2 = r_series.shift(2)
    r_lag3 = r_series.shift(3)
    
    # Rolling mean 3 và 6 của r_{t-1}
    # shift(1) trước khi rolling đảm bảo window lấy r_{t-1}...r_{t-k}
    r_roll3 = r_series.shift(1).rolling(window=3).mean()
    r_roll6 = r_series.shift(1).rolling(window=6).mean()
    
    df_features = pd.DataFrame({
        "r_lag1": r_lag1,
        "r_lag2": r_lag2,
        "r_lag3": r_lag3,
        "r_roll3": r_roll3,
        "r_roll6": r_roll6,
    }, index=r_series.index)
    
    return df_features


def build_supervised_data(y: pd.Series, dropna: bool = True) -> pd.DataFrame:
    """
    Chuyển đổi chuỗi gốc y_t thành bộ dữ liệu học có giám sát (Supervised Learning Dataset).
    Cột đầu ra:
        - y: giá trị thực tế tại thời điểm t
        - r: biến mục tiêu (target) tại thời điểm t
        - r_lag1, r_lag2, r_lag3, r_roll3, r_roll6: các đặc trưng đầu vào X_t
    """
    r = compute_log_growth(y)
    X = create_features(r)
    
    df_sup = pd.DataFrame({
        "y": y,
        "r": r,
    }, index=y.index)
    
    df_sup = pd.concat([df_sup, X], axis=1)
    
    if dropna:
        df_sup = df_sup.dropna().copy()
        
    return df_sup


def compute_step_features_from_buffer(r_buffer: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Tính nhanh 5 đặc trưng cho bước dự báo tiếp theo từ một buffer chứa các giá trị r lịch sử.
    Yêu cầu buffer có ít nhất 6 giá trị r gần nhất (từ t-5 đến t).
    Giá trị cuối cùng của buffer r_buffer[-1] đóng vai trò là r_{t}.
    Khi dự báo cho bước tiếp theo (t+1), các đặc trưng là:
        - lag1: r_buffer[-1]
        - lag2: r_buffer[-2]
        - lag3: r_buffer[-3]
        - roll3: mean(r_buffer[-3:])
        - roll6: mean(r_buffer[-6:])
    Hàm này được dùng trong vòng lặp dự báo đệ quy 12 bước.
    """
    buf = list(r_buffer)
    if len(buf) < 6:
        raise ValueError(f"Buffer cần tối thiểu 6 giá trị r để tính rolling 6, hiện có: {len(buf)}")
        
    lag1 = buf[-1]
    lag2 = buf[-2]
    lag3 = buf[-3]
    roll3 = float(np.mean(buf[-3:]))
    roll6 = float(np.mean(buf[-6:]))
    
    return np.array([lag1, lag2, lag3, roll3, roll6], dtype=float)


def get_feature_names() -> List[str]:
    """Trả về danh sách 5 tên đặc trưng chuẩn."""
    return list(FEATURE_NAMES)


def get_feature_dictionary(sample_row_features: Dict[str, float] = None) -> pd.DataFrame:
    """
    Trả về DataFrame từ điển đặc trưng kèm ví dụ mẫu số liệu cụ thể.
    """
    rows = []
    for item in FEATURE_METADATA:
        row = dict(item)
        if sample_row_features and row["feature_name"] in sample_row_features:
            row["sample_value_2017_08"] = sample_row_features[row["feature_name"]]
        rows.append(row)
    return pd.DataFrame(rows)
