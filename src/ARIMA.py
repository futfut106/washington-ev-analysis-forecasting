"""
ARIMA.py - Xây dựng và đánh giá mô hình ARIMA(2,2,2) cho chuỗi EV Total tại Washington
Môn: Phân tích dữ liệu kinh doanh (IS403.R14) - Nhóm 8
Chương 4: Phân tích chuỗi thời gian và dự báo thống kê (Mục 4.4)

Tập dữ liệu: Electric_Vehicle_Population_Data.csv (Dataset [H])
- Train: 71 tháng (01/2017 - 11/2022)
- Test : 12 tháng (12/2022 - 11/2023)
- Cấu hình chốt: ARIMA(2,2,2) không hằng số (trend='n')
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# Thiết lập đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "arima_out"
FIGURES_DIR = BASE_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_TRAIN = 71
N_TEST = 12


def load_and_aggregate_data():
    """Tải dataset [H], tổng hợp EV Total toàn bang theo tháng và xuất ch4_series.csv"""
    data_path = DATA_DIR / "Electric_Vehicle_Population_Data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

    df = pd.read_csv(data_path)
    df.columns = [
        "Date", "County", "State", "Vehicle_Use", 
        "BEV", "PHEV", "EV_Total", "Non_EV", "Total_Vehicles", "Percent_EV"
    ]
    df["Date"] = pd.to_datetime(df["Date"])

    # Tổng hợp toàn bang theo tháng
    monthly = df.groupby("Date").agg({
        "BEV": "sum",
        "PHEV": "sum",
        "EV_Total": "sum",
        "Non_EV": "sum",
        "Total_Vehicles": "sum"
    }).sort_index()

    monthly["ev_pct"] = (monthly["EV_Total"] / monthly["Total_Vehicles"]) * 100
    monthly["split"] = ["train" if i < N_TRAIN else "test" for i in range(len(monthly))]
    
    # Lưu ch4_series.csv phục vụ đầu vào chuẩn hóa cho toàn bộ Chương 4
    ch4_series_path = DATA_DIR / "ch4_series.csv"
    monthly.reset_index().rename(columns={"Date": "date", "BEV": "bev", "PHEV": "phev", 
                                          "EV_Total": "ev_total", "Non_EV": "non_ev", 
                                          "Total_Vehicles": "total_vehicles"}).to_csv(ch4_series_path, index=False)
    print(f"[OK] Đã lưu chuỗi chuẩn hóa: {ch4_series_path} ({len(monthly)} tháng)")

    series = monthly["EV_Total"].astype(float)
    series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp("M")
    series = series.asfreq("ME")
    return series


def calculate_metrics(actual, forecast):
    """Tính toán bộ chỉ số đánh giá sai số chuẩn mực"""
    errors = actual - forecast
    abs_errors = np.abs(errors)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(abs_errors / actual) * 100
    smape = np.mean(2 * abs_errors / (np.abs(actual) + np.abs(forecast))) * 100
    bias = np.mean(errors)
    max_err_idx = np.argmax(abs_errors)
    max_err_val = abs_errors[max_err_idx]
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "sMAPE (%)": round(smape, 2),
        "Mean_Bias": round(bias, 2),
        "Max_Abs_Error": round(max_err_val, 2),
        "Max_Err_Month": max_err_idx
    }


def fit_arima(train_series, order=(2, 2, 2)):
    """Ước lượng mô hình ARIMA(2,2,2) với trend='n'"""
    print(f"\n[INFO] Đang ước lượng ARIMA{order} trên {len(train_series)} tháng Train...")
    model = ARIMA(train_series, order=order, trend="n")
    res = model.fit()
    print(res.summary())
    return res


def evaluate_and_plot(res, train, test):
    """Dự báo 12 tháng Test, chẩn đoán phần dư và lưu kết quả"""
    # 1. Dự báo ngoài mẫu (Out-of-sample)
    forecast_res = res.get_forecast(steps=N_TEST)
    forecast_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)  # Khoảng dự báo 95% (Prediction Interval)
    
    metrics = calculate_metrics(test.values, forecast_mean.values)
    print("\n--- KẾT QUẢ ĐÁNH GIÁ ARIMA(2,2,2) TRÊN TẬP TEST (12 THÁNG) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Tạo bảng chi tiết
    df_forecast = pd.DataFrame({
        "Date": test.index.strftime("%Y-%m"),
        "Actual": test.values,
        "Forecast": np.round(forecast_mean.values, 2),
        "Error": np.round(test.values - forecast_mean.values, 2),
        "Abs_Pct_Error (%)": np.round(np.abs(test.values - forecast_mean.values) / test.values * 100, 2),
        "Lower_95": np.round(conf_int.iloc[:, 0].values, 2),
        "Upper_95": np.round(conf_int.iloc[:, 1].values, 2)
    })
    
    csv_out = RESULTS_DIR / "arima_222_forecast_test.csv"
    df_forecast.to_csv(csv_out, index=False)
    print(f"[OK] Đã lưu chi tiết dự báo tại: {csv_out}")

    # 2. Chẩn đoán phần dư
    residuals = res.resid.iloc[order_d(res.model.order):]
    lb_test = acorr_ljungbox(residuals, lags=[12, 24], model_df=sum(res.model.order[::2]))
    jb_stat, jb_p = stats.jarque_bera(residuals)
    
    diag_summary = {
        "AIC": round(res.aic, 2),
        "BIC": round(res.bic, 2),
        "LjungBox_p12": round(lb_test["lb_pvalue"].iloc[0], 4),
        "LjungBox_p24": round(lb_test["lb_pvalue"].iloc[1], 4),
        "JarqueBera_stat": round(jb_stat, 4),
        "JarqueBera_p": round(jb_p, 4)
    }
    pd.DataFrame([diag_summary]).to_csv(RESULTS_DIR / "arima_222_diagnostics.csv", index=False)

    # 3. Vẽ biểu đồ dự báo
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train.index[-24:], train.values[-24:], label="Train (24 tháng cuối)", color="#1f77b4", lw=1.8)
    ax.plot(test.index, test.values, label="Thực tế (Test)", color="black", lw=2, marker="o", markersize=4)
    ax.plot(test.index, forecast_mean.values, label="Dự báo ARIMA(2,2,2)", color="#d62728", lw=2, ls="--", marker="s", markersize=4)
    ax.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="#d62728", alpha=0.15, label="Khoảng dự báo 95%")
    
    ax.set_title("Dự báo EV Total tại Washington bằng ARIMA(2,2,2) trên 12 tháng Test", fontsize=12, fontweight="bold")
    ax.set_xlabel("Thời gian (tháng)", fontsize=10)
    ax.set_ylabel("Số lượng xe (EV Total)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    
    fig_path = FIGURES_DIR / "arima_222_forecast.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Đã lưu biểu đồ dự báo tại: {fig_path}")


def order_d(order):
    return order[1]


def main():
    series = load_and_aggregate_data()
    train, test = series.iloc[:N_TRAIN], series.iloc[N_TRAIN:]
    assert len(train) == N_TRAIN and len(test) == N_TEST
    
    model_res = fit_arima(train, order=(2, 2, 2))
    evaluate_and_plot(model_res, train, test)
    print("\n[HOÀN THÀNH] Đã hoàn tất thực nghiệm mô hình ARIMA(2,2,2).")


if __name__ == "__main__":
    main()

