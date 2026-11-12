"""
arima_compare.py - So sánh tổng hợp hiệu năng các mô hình chuỗi thời gian thống kê
Môn: Phân tích dữ liệu kinh doanh (IS403.R14) - Nhóm 8
Chương 4: Phân tích chuỗi thời gian và dự báo thống kê (Mục 4.6 & Bảng 4.9, Hình 4.9)

So sánh:
1. Naive Baseline
2. Seasonal Naive Baseline
3. ARIMA(2,2,2) (Phương án A chuẩn hóa)
4. SARIMA A: (1,1,1) x (1,1,0)_12
5. SARIMA B: (0,1,1) x (1,1,1)_12
6. SARIMA C: (0,2,1) x (0,1,1)_12

Tập dữ liệu: 83 tháng (01/2017 - 11/2023)
- Train: 71 tháng (01/2017 - 11/2022)
- Test : 12 tháng (12/2022 - 11/2023)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SARIMA_OUT = RESULTS_DIR / "sarima_out"
FIGURES_DIR = BASE_DIR / "figures"

SARIMA_OUT.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_TRAIN = 71
N_TEST = 12
SEASONAL_PERIOD = 12


def load_series():
    """Tải chuỗi EV_Total đã được tổng hợp toàn bang"""
    ch4_series_path = DATA_DIR / "ch4_series.csv"
    if ch4_series_path.exists():
        df = pd.read_csv(ch4_series_path)
        series = df["ev_total"].astype(float)
        series.index = pd.to_datetime(df["date"])
    else:
        # Nếu chưa có ch4_series.csv thì đọc từ file gốc và tạo
        from ARIMA import load_and_aggregate_data
        series = load_and_aggregate_data()
    
    series = series.asfreq("ME")
    return series


def compute_metrics(actual, pred):
    """Tính toán bộ chỉ số MAE, RMSE, MAPE, sMAPE, Bias"""
    err = actual - pred
    abs_err = np.abs(err)
    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean(err ** 2))
    mape = np.mean(abs_err / actual) * 100
    smape = np.mean(2 * abs_err / (np.abs(actual) + np.abs(pred))) * 100
    bias = np.mean(err)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "sMAPE (%)": smape,
        "Bias": bias
    }


def run_all_models(train, test):
    """Huấn luyện và dự báo tất cả các mô hình trên cùng 12 tháng Test"""
    forecasts = {}
    intervals = {}
    
    # 1. Naive Baseline (Lấy giá trị cuối của Train)
    naive_val = train.iloc[-1]
    forecasts["Naive"] = np.repeat(naive_val, N_TEST)
    intervals["Naive"] = None
    
    # 2. Seasonal Naive Baseline (Lấy 12 tháng cuối của Train chiếu tương ứng)
    snaive_vals = train.iloc[-SEASONAL_PERIOD:].values
    forecasts["Seasonal_Naive"] = snaive_vals
    intervals["Seasonal_Naive"] = None
    
    # 3. ARIMA(2,2,2) - Chuẩn hóa theo Phương án A
    arima_mod = ARIMA(train, order=(2, 2, 2), trend="n").fit()
    arima_res = arima_mod.get_forecast(steps=N_TEST)
    forecasts["ARIMA(2,2,2)"] = arima_res.predicted_mean.values
    intervals["ARIMA(2,2,2)"] = arima_res.conf_int(alpha=0.05).values
    
    # 4. SARIMA Configurations
    sarima_configs = {
        "SARIMA_A": ((1, 1, 1), (1, 1, 0, SEASONAL_PERIOD)),
        "SARIMA_B": ((0, 1, 1), (1, 1, 1, SEASONAL_PERIOD)),
        "SARIMA_C": ((0, 2, 1), (0, 1, 1, SEASONAL_PERIOD)),
    }
    
    for name, (order, s_order) in sarima_configs.items():
        s_mod = SARIMAX(train, order=order, seasonal_order=s_order, 
                        trend="n", enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=500)
        s_res = s_mod.get_forecast(steps=N_TEST)
        forecasts[name] = s_res.predicted_mean.values
        intervals[name] = s_res.conf_int(alpha=0.05).values
        
    return forecasts, intervals


def build_comparison_table(test, forecasts):
    """Tổng hợp Bảng 4.9: So sánh hiệu năng và % cải thiện RMSE"""
    rows = []
    actual = test.values
    
    naive_rmse = np.sqrt(np.mean((actual - forecasts["Naive"]) ** 2))
    arima_rmse = np.sqrt(np.mean((actual - forecasts["ARIMA(2,2,2)"]) ** 2))
    
    model_labels = {
        "Naive": "Naive Baseline",
        "Seasonal_Naive": "Seasonal Naive",
        "ARIMA(2,2,2)": "ARIMA(2,2,2) (Mục 4.4)",
        "SARIMA_A": "SARIMA A (1,1,1)x(1,1,0)12",
        "SARIMA_B": "SARIMA B (0,1,1)x(1,1,1)12",
        "SARIMA_C": "SARIMA C (0,2,1)x(0,1,1)12"
    }
    
    for key, label in model_labels.items():
        pred = forecasts[key]
        m = compute_metrics(actual, pred)
        
        # % Cải thiện RMSE so với Naive
        pct_imp_naive = ((naive_rmse - m["RMSE"]) / naive_rmse) * 100 if key != "Naive" else 0.0
        
        # % Cải thiện RMSE so với ARIMA
        if key in ["Naive", "Seasonal_Naive"]:
            pct_imp_arima = np.nan
        elif key == "ARIMA(2,2,2)":
            pct_imp_arima = 0.0
        else:
            pct_imp_arima = ((arima_rmse - m["RMSE"]) / arima_rmse) * 100
            
        rows.append({
            "Mô hình": label,
            "MAE (xe)": round(m["MAE"], 2),
            "RMSE (xe)": round(m["RMSE"], 2),
            "MAPE (%)": f"{m['MAPE (%)']:.2f}%",
            "sMAPE (%)": f"{m['sMAPE (%)']:.2f}%",
            "Bias (xe)": round(m["Bias"], 2),
            "Cải thiện vs Naive (%)": f"{pct_imp_naive:+.2f}%" if key != "Naive" else "Gốc so sánh",
            "Cải thiện vs ARIMA (%)": f"{pct_imp_arima:+.2f}%" if not np.isnan(pct_imp_arima) and key != "ARIMA(2,2,2)" else ("Gốc so sánh" if key == "ARIMA(2,2,2)" else "—")
        })
        
    df_comp = pd.DataFrame(rows)
    return df_comp


def plot_forecast_comparison(train, test, forecasts, intervals):
    """
    Vẽ Hình 4.9: So sánh thực tế và dự báo của các mô hình trên 12 tháng Test
    Đáp ứng yêu cầu chèn hình thiếu trong báo cáo đồ án.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Vẽ chuỗi train 18 tháng cuối
    history_win = train.iloc[-18:]
    ax.plot(history_win.index, history_win.values / 1000, color="gray", lw=1.8, label="Thực tế Train (18 tháng cuối)")
    
    # Vẽ thực tế test
    ax.plot(test.index, test.values / 1000, color="black", lw=2.5, marker="o", markersize=5, label="Thực tế Test (12 tháng)", zorder=10)
    
    # Đường phân định Train - Test
    ax.axvline(train.index[-1], color="red", linestyle=":", lw=1.5, alpha=0.8, label="Mốc chia Train/Test (11/2022)")
    
    # Dự báo các mô hình
    styles = {
        "Naive": ("#7f7f7f", ":", 1.5),
        "Seasonal_Naive": ("#bcbd22", "-.", 1.5),
        "ARIMA(2,2,2)": ("#1f77b4", "--", 2.0),
        "SARIMA_A": ("#2ca02c", "--", 1.8),
        "SARIMA_C": ("#d62728", "-", 2.5),
    }
    
    for key, (col, ls, lw) in styles.items():
        pred = forecasts[key]
        ax.plot(test.index, pred / 1000, color=col, linestyle=ls, linewidth=lw, label=f"Dự báo {key}")
        
    # Khoảng dự báo 95% của mô hình tốt nhất (SARIMA C)
    if intervals["SARIMA_C"] is not None:
        ci = intervals["SARIMA_C"]
        ax.fill_between(test.index, ci[:, 0] / 1000, ci[:, 1] / 1000, color="#d62728", alpha=0.15, label="Khoảng dự báo 95% (SARIMA C)")
        
    ax.set_title("Hình 4.9: Giá trị thực tế và dự báo EV Total trên 12 tháng test từ mốc 11/2022", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Thời gian (tháng)", fontsize=11)
    ax.set_ylabel("Quy mô xe điện (nghìn xe)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    
    fig_path = FIGURES_DIR / "forecast_comparison.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=250)
    plt.savefig(SARIMA_OUT / "forecast_comparison.png", dpi=250)
    plt.close()
    print(f"[OK] Đã lưu Hình 4.9 (forecast_comparison.png) tại: {fig_path}")


def main():
    print("=== CHƯƠNG TRÌNH SO SÁNH TỔNG HỢP MÔ HÌNH THỜI GIAN (ARIMA & SARIMA) ===")
    series = load_series()
    train, test = series.iloc[:N_TRAIN], series.iloc[N_TRAIN:]
    print(f"Tổng quan: Train = {len(train)} tháng ({train.index.min():%m/%Y} -> {train.index.max():%m/%Y}) | Test = {len(test)} tháng")
    
    # Chạy tất cả mô hình
    forecasts, intervals = run_all_models(train, test)
    
    # Bảng so sánh
    df_comparison = build_comparison_table(test, forecasts)
    print("\n--- BẢNG 4.9: TỔNG HỢP SO SÁNH HIỆU NĂNG TRÊN 12 THÁNG TEST ---")
    print(df_comparison.to_string(index=False))
    
    # Lưu CSV
    out_csv = RESULTS_DIR / "model_comparison_summary.csv"
    df_comparison.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_comparison.to_csv(SARIMA_OUT / "sarima_comparison_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n[OK] Đã xuất kết quả so sánh tại: {out_csv}")
    
    # Vẽ Hình 4.9
    plot_forecast_comparison(train, test, forecasts, intervals)
    print("\n[HOÀN TẤT] Đồng bộ dữ liệu, mô hình và biểu đồ thành công!")


if __name__ == "__main__":
    main()

