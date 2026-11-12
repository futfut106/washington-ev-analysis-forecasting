"""
So sánh chẩn đoán mô hình ARIMA cho chuỗi EV Washington (TV3 Tuấn Anh).
Kiểm tra tiêu chí thông tin AIC, BIC và kiểm định tự tương quan phần dư Ljung-Box.
"""
from pathlib import Path
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


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


def main():
    data_path = find_data_file()
    print(f"Đọc dữ liệu từ: {data_path}")
    df = pd.read_csv(data_path)
    
    target_col = "EV_Total" if "EV_Total" in df.columns else "ev_total"
    train = df.iloc[:-12].copy()

    orders = [(2, 2, 2), (2, 2, 0), (0, 2, 1)]

    for order in orders:
        print(f"\n===== ARIMA{order} =====")

        model = ARIMA(train[target_col], order=order)
        fit = model.fit()

        burn = max(fit.loglikelihood_burn, 2)
        residuals = fit.resid[burn:]

        lb = acorr_ljungbox(
            residuals,
            lags=[10, 15],
            model_df=order[0] + order[2],
            return_df=True
        )

        print(f"AIC: {fit.aic:.4f}")
        print(f"BIC: {fit.bic:.4f}")
        print("Ljung-Box test:")
        print(lb)
        print("Hội tụ:", fit.mle_retvals.get("converged", None))


if __name__ == "__main__":
    main()
