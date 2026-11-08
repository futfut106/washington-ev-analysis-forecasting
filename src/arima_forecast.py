import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Đọc dữ liệu
df = pd.read_csv("EV_ARIMA_Monthly.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Chia train và test
train = df.iloc[:-12]
test = df.iloc[-12:]

# Huấn luyện mô hình ARIMA(2,2,2)
model = ARIMA(train["EV_Total"], order=(2, 2, 2))
fit = model.fit()

# Dự báo 12 tháng
forecast = fit.forecast(steps=len(test))

# Tạo bảng kết quả
result = pd.DataFrame({
    "Date": test["Date"].values,
    "Actual": test["EV_Total"].values,
    "Forecast": forecast.values
})

print(result)

# Đánh giá mô hình
mae = mean_absolute_error(result["Actual"], result["Forecast"])
rmse = np.sqrt(mean_squared_error(result["Actual"], result["Forecast"]))
mape = np.mean(
    np.abs((result["Actual"] - result["Forecast"]) / result["Actual"])
) * 100

print("\nMAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
plt.plot(train["Date"], train["EV_Total"], label="Training Data")
plt.plot(test["Date"], test["EV_Total"], label="Actual")
plt.plot(test["Date"], forecast.values, label="Forecast")
plt.title("ARIMA(2,2,2) - Actual vs Forecast")
plt.xlabel("Date")
plt.ylabel("EV Total")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()