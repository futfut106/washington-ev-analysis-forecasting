"""
Mô hình dự báo ARIMA(2,2,2) cho quy mô xe điện Washington (Mục 4.4 / TV3 Tuấn Anh).
Huấn luyện trên 71 tháng train (01/2017 - 11/2022) và dự báo 12 tháng test (12/2022 - 11/2023).
Xuất kết quả chi tiết ra results/forecast_arima.csv và kết quả đánh giá ra results/arima_test_metrics.json.
"""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


def find_data_file() -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    candidates = [
        base_dir / "data" / "EV_ARIMA_Monthly.csv",
        base_dir / "data" / "ch4_series.csv",
        Path("data/EV_ARIMA_Monthly.csv"),
        Path("EV_ARIMA_Monthly.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Không tìm thấy file dữ liệu EV_ARIMA_Monthly.csv hoặc ch4_series.csv.")


def run_forecast(show_plot: bool = False):
    data_path = find_data_file()
    print(f"Đọc dữ liệu từ: {data_path}")
    df = pd.read_csv(data_path)
    
    # Chuẩn hoá tên cột
    date_col = "Date" if "Date" in df.columns else "date"
    target_col = "EV_Total" if "EV_Total" in df.columns else "ev_total"
    
    df["Date"] = pd.to_datetime(df[date_col])
    df["EV_Total"] = df[target_col].astype(float)
    df = df.sort_values("Date").reset_index(drop=True)

    # Chia train (71 tháng) và test (12 tháng)
    train = df.iloc[:-12].copy()
    test = df.iloc[-12:].copy()

    origin_date = train["Date"].iloc[-1].strftime("%Y-%m-%d")
    print(f"Khoảng thời gian Train: {train['Date'].min().strftime('%Y-%m-%d')} đến {origin_date} ({len(train)} tháng)")
    print(f"Khoảng thời gian Test : {test['Date'].min().strftime('%Y-%m-%d')} đến {test['Date'].iloc[-1].strftime('%Y-%m-%d')} ({len(test)} tháng)")

    # Huấn luyện mô hình ARIMA(2,2,2)
    print("\nĐang huấn luyện ARIMA(2,2,2)...")
    model = ARIMA(train["EV_Total"], order=(2, 2, 2))
    fit = model.fit()
    print(f"Hội tụ: {fit.mle_retvals.get('converged', True)} | AIC: {fit.aic:.2f} | BIC: {fit.bic:.2f}")

    # Dự báo 12 tháng test
    forecast_values = fit.forecast(steps=len(test))

    # Bảng kết quả dự báo
    dates_str = [d.strftime("%Y-%m-%d") for d in test["Date"]]
    actual_vals = test["EV_Total"].values
    pred_vals = forecast_values.values

    # Tính toán các chỉ số đánh giá
    errors = actual_vals - pred_vals
    mae = mean_absolute_error(actual_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(actual_vals, pred_vals))
    mape = np.mean(np.abs(errors / actual_vals)) * 100
    denom = np.abs(actual_vals) + np.abs(pred_vals)
    smape = 100.0 * np.mean(np.divide(2.0 * np.abs(errors), denom, out=np.zeros_like(denom), where=denom != 0))
    bias = float(np.mean(errors))

    print("\n===== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (12 THÁNG) =====")
    print(f"MAE  : {mae:.5f}")
    print(f"RMSE : {rmse:.5f}")
    print(f"MAPE : {mape:.5f}%")
    print(f"sMAPE: {smape:.5f}%")
    print(f"Bias : {bias:.5f}")

    # Tạo DataFrame dự báo chuẩn hoá theo schema Chương 5
    forecast_df = pd.DataFrame({
        "model": "ARIMA(2,2,2)",
        "date": dates_str,
        "origin": origin_date,
        "horizon": list(range(1, len(test) + 1)),
        "actual": [int(x) for x in actual_vals],
        "pred": pred_vals,
        "config_id": "arima_2_2_2",
        "target_transform": "level"
    })

    print("\nChi tiết 12 tháng dự báo:")
    print(forecast_df[["date", "horizon", "actual", "pred"]].to_string(index=False))

    # Xuất ra thư mục results
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results"
    figures_dir = base_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_out = results_dir / "forecast_arima.csv"
    forecast_df.to_csv(csv_out, index=False)
    print(f"\n[OK] Đã lưu dự báo chi tiết: {csv_out}")

    metrics_out = results_dir / "arima_test_metrics.json"
    metrics_data = {
        "model": "ARIMA(2,2,2)",
        "order": [2, 2, 2],
        "origin": origin_date,
        "horizon": 12,
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "smape": float(smape),
        "bias": float(bias),
        "aic": float(fit.aic),
        "bic": float(fit.bic)
    }
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OK] Đã lưu chỉ số đánh giá: {metrics_out}")

    # Vẽ và lưu biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train["Date"], train["EV_Total"], label="Lịch sử (Train)", color="#1f77b4", lw=2)
    ax.plot(test["Date"], test["EV_Total"], label="Thực tế (Test)", color="#2ca02c", lw=2, marker="o")
    ax.plot(test["Date"], pred_vals, label="Dự báo ARIMA(2,2,2)", color="#d62728", lw=2, linestyle="--", marker="s")
    ax.set_title("ARIMA(2,2,2) - Washington EV Total Forecast (2022-12 đến 2023-11)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Thời gian", fontsize=11)
    ax.set_ylabel("Số lượng xe điện (EV Total)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    fig.tight_layout()

    fig_out = figures_dir / "ARIMA_Forecast_12M.png"
    fig.savefig(fig_out, dpi=300)
    print(f"[OK] Đã lưu biểu đồ: {fig_out}")

    if show_plot:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự báo xe điện Washington với ARIMA(2,2,2)")
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ đồ thị")
    args = parser.parse_args()
    run_forecast(show_plot=args.show)
