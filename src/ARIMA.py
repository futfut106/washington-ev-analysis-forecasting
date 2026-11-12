"""
Quy trình xây dựng và lựa chọn mô hình ARIMA cho chuỗi xe điện Washington (TV3 Tuấn Anh).
Bao gồm:
1. Đọc và kiểm tra tính dừng của chuỗi bằng kiểm định ADF (gốc, sai phân bậc 1, sai phân bậc 2).
2. Vẽ biểu đồ ACF/PACF trên chuỗi sai phân bậc 2.
3. Huấn luyện lưới các mô hình ARIMA(p, 2, q), so sánh theo tiêu chí AIC/BIC.
4. Xuất kết quả so sánh vào results/ARIMA_Comparison.csv.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


def load_monthly_series() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    # Ưu tiên file chuỗi tháng sẵn có
    for p in [data_dir / "EV_ARIMA_Monthly.csv", data_dir / "ch4_series.csv", Path("EV_ARIMA_Monthly.csv")]:
        if p.exists():
            df = pd.read_csv(p)
            date_col = "Date" if "Date" in df.columns else "date"
            val_col = "EV_Total" if "EV_Total" in df.columns else "ev_total"
            out = pd.DataFrame({
                "Date": pd.to_datetime(df[date_col]),
                "EV_Total": df[val_col].astype(float)
            }).sort_values("Date").reset_index(drop=True)
            return out

    # Nếu chưa có file tháng, tổng hợp từ file lịch sử thô
    raw_candidates = [
        Path("EV_Population_History_Washington_Clean.csv"),
        data_dir / "Electric_Vehicle_Population_Data.csv",
        data_dir / "EV_Population_Washington_Clean__1_.csv"
    ]
    for p in raw_candidates:
        if p.exists():
            df = pd.read_csv(p)
            date_col = "Date" if "Date" in df.columns else "date"
            val_col = "Electric Vehicle (EV) Total" if "Electric Vehicle (EV) Total" in df.columns else "ev_total"
            df[date_col] = pd.to_datetime(df[date_col])
            monthly = df.groupby(date_col, as_index=False)[val_col].sum()
            monthly.columns = ["Date", "EV_Total"]
            monthly = monthly.sort_values("Date").reset_index(drop=True)
            monthly.to_csv(data_dir / "EV_ARIMA_Monthly.csv", index=False)
            return monthly

    raise FileNotFoundError("Không tìm thấy dữ liệu đầu vào phù hợp.")


def main(interactive: bool = False):
    base_dir = Path(__file__).resolve().parent.parent
    figures_dir = base_dir / "figures"
    results_dir = base_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    monthly_data = load_monthly_series()
    print("Dữ liệu đầu vào:")
    print(monthly_data.head())
    print("Kích thước:", monthly_data.shape)

    # Chia dữ liệu Train/Test (71 tháng train, 12 tháng test)
    train = monthly_data.iloc[:-12]
    test = monthly_data.iloc[-12:]

    print("\nSố quan sát Train:", len(train))
    print("Số quan sát Test:", len(test))
    print("Thời gian Train:", train["Date"].min().strftime("%Y-%m-%d"), "đến", train["Date"].max().strftime("%Y-%m-%d"))
    print("Thời gian Test :", test["Date"].min().strftime("%Y-%m-%d"), "đến", test["Date"].max().strftime("%Y-%m-%d"))

    # Kiểm định ADF trên dữ liệu Train
    result = adfuller(train["EV_Total"])
    print("\n===== KIỂM ĐỊNH ADF GỐC =====")
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    print("Kết luận:", "Dừng" if result[1] <= 0.05 else "Chưa dừng (cần sai phân)")

    # Sai phân bậc 1 trên tập Train
    train_diff = train["EV_Total"].diff().dropna()
    result_diff = adfuller(train_diff, autolag="AIC")
    print("\n===== ADF SAU SAI PHÂN BẬC 1 =====")
    print("ADF Statistic:", result_diff[0])
    print("p-value:", result_diff[1])
    print("Critical Values:", result_diff[4])
    print("Kết luận:", "Dừng" if result_diff[1] <= 0.05 else "Chưa dừng")

    # Sai phân bậc 2
    train_diff2 = train["EV_Total"].diff().diff().dropna()
    result_diff2 = adfuller(train_diff2, autolag="AIC")
    print("\n===== ADF SAU SAI PHÂN BẬC 2 =====")
    print("ADF Statistic:", result_diff2[0])
    print("p-value:", result_diff2[1])
    print("Critical Values:", result_diff2[4])
    print("Kết luận:", "Dừng" if result_diff2[1] <= 0.05 else "Chưa dừng")

    # Biểu đồ ACF
    fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
    plot_acf(train_diff2, lags=20, ax=ax_acf)
    ax_acf.set_title("ACF - Sai phân bậc 2")
    fig_acf.tight_layout()
    fig_acf.savefig(figures_dir / "ACF_ARIMA.png", dpi=300)
    if interactive:
        plt.show()
    plt.close(fig_acf)

    # Biểu đồ PACF
    fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
    plot_pacf(train_diff2, lags=20, method="ywm", ax=ax_pacf)
    ax_pacf.set_title("PACF - Sai phân bậc 2")
    fig_pacf.tight_layout()
    fig_pacf.savefig(figures_dir / "PACF_ARIMA.png", dpi=300)
    if interactive:
        plt.show()
    plt.close(fig_pacf)

    # Danh sách mô hình ARIMA thử nghiệm
    orders = [
        (1, 2, 0),
        (2, 2, 0),
        (0, 2, 1),
        (1, 2, 1),
        (2, 2, 1),
        (2, 2, 2)
    ]

    results = []
    print("\n===== HUẤN LUYỆN VÀ SO SÁNH CÁC MÔ HÌNH ARIMA =====")
    for order in orders:
        try:
            model = ARIMA(train["EV_Total"], order=order)
            fitted_model = model.fit()

            results.append({
                "Model": f"ARIMA{order}",
                "AIC": fitted_model.aic,
                "BIC": fitted_model.bic
            })

            print(f"ARIMA{order} - AIC: {fitted_model.aic:.4f}, BIC: {fitted_model.bic:.4f}")

        except Exception as e:
            print(f"ARIMA{order} gặp lỗi: {e}")

    # Sắp xếp theo AIC tăng dần
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(by="AIC").reset_index(drop=True)

    print("\n===== BẢNG SO SÁNH MÔ HÌNH =====")
    print(comparison.to_string(index=False))

    comparison_path = results_dir / "ARIMA_Comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\n[OK] Đã lưu bảng so sánh: {comparison_path}")


if __name__ == "__main__":
    main()
