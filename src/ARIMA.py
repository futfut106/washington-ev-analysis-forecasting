import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("EV_Population_History_Washington_Clean.csv")

# Chuyển Date sang kiểu datetime
df["Date"] = pd.to_datetime(df["Date"])

# Tổng hợp tổng số xe điện theo tháng
monthly_data = df.groupby("Date", as_index=False)[
    "Electric Vehicle (EV) Total"
].sum()

# Đổi tên cột
monthly_data.columns = ["Date", "EV_Total"]

# Xuất dữ liệu
monthly_data.to_csv("EV_ARIMA_Monthly.csv", index=False)

# Kiểm tra
print(monthly_data.head())
print(monthly_data.shape)


# Chia dữ liệu Train/Test
train = monthly_data.iloc[:-12]
test = monthly_data.iloc[-12:]

print("Số quan sát Train:", len(train))
print("Số quan sát Test:", len(test))

print("\nThời gian Train:")
print(train["Date"].min(), "đến", train["Date"].max())

print("\nThời gian Test:")
print(test["Date"].min(), "đến", test["Date"].max())


from statsmodels.tsa.stattools import adfuller

# Kiểm định ADF trên dữ liệu Train
result = adfuller(train["EV_Total"])

print("\n===== KIỂM ĐỊNH ADF =====")
print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:", result[4])

if result[1] <= 0.05:
    print("Kết luận: Chuỗi dữ liệu dừng.")
else:
    print("Kết luận: Chuỗi dữ liệu chưa dừng.")


# Sai phân bậc 1 trên tập Train
train_diff = train["EV_Total"].diff().dropna()

# Kiểm định ADF lần 2
result_diff = adfuller(train_diff, autolag="AIC")

print("\n===== ADF SAU SAI PHÂN BẬC 1 =====")
print("ADF Statistic:", result_diff[0])
print("p-value:", result_diff[1])
print("Critical Values:", result_diff[4])

if result_diff[1] <= 0.05:
    print("Kết luận: Chuỗi sau sai phân đã dừng.")
else:
    print("Kết luận: Chuỗi sau sai phân vẫn chưa dừng.")


# Sai phân bậc 2
train_diff2 = train["EV_Total"].diff().diff().dropna()

# Kiểm định ADF lần 3
result_diff2 = adfuller(train_diff2, autolag="AIC")

print("\n===== ADF SAU SAI PHÂN BẬC 2 =====")
print("ADF Statistic:", result_diff2[0])
print("p-value:", result_diff2[1])
print("Critical Values:", result_diff2[4])

if result_diff2[1] <= 0.05:
    print("Kết luận: Chuỗi sau sai phân bậc 2 đã dừng.")
else:
    print("Kết luận: Chuỗi sau sai phân bậc 2 vẫn chưa dừng.")

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Biểu đồ ACF
fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(train_diff2, lags=20, ax=ax)
ax.set_title("ACF - Sai phân bậc 2")
fig.tight_layout()
fig.savefig("ACF_ARIMA.png", dpi=300)
plt.show()
plt.close(fig)

# Biểu đồ PACF
fig, ax = plt.subplots(figsize=(10, 4))
plot_pacf(train_diff2, lags=20, method="ywm", ax=ax)
ax.set_title("PACF - Sai phân bậc 2")
fig.tight_layout()
fig.savefig("PACF_ARIMA.png", dpi=300)
plt.show()
plt.close(fig)


from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

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

for order in orders:
    try:
        model = ARIMA(train["EV_Total"], order=order)
        fitted_model = model.fit()

        results.append({
            "Model": f"ARIMA{order}",
            "AIC": fitted_model.aic,
            "BIC": fitted_model.bic
        })

        print(f"ARIMA{order} - AIC: {fitted_model.aic:.4f}, "
              f"BIC: {fitted_model.bic:.4f}")

    except Exception as e:
        print(f"ARIMA{order} gặp lỗi: {e}")

# Sắp xếp theo AIC tăng dần
comparison = pd.DataFrame(results)
comparison = comparison.sort_values(by="AIC")

print("\n===== BẢNG SO SÁNH MÔ HÌNH =====")
print(comparison.to_string(index=False))

comparison.to_csv("ARIMA_Comparison.csv", index=False)