
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

df = pd.read_csv("EV_ARIMA_Monthly.csv")
train = df.iloc[:-12]

orders = [(2, 2, 2), (2, 2, 0), (0, 2, 1)]

for order in orders:
    print(f"\n===== ARIMA{order} =====")

    model = ARIMA(train["EV_Total"], order=order)
    fit = model.fit()

    burn = max(fit.loglikelihood_burn, 2)
    residuals = fit.resid[burn:]

    lb = acorr_ljungbox(
        residuals,
        lags=[10, 15],
        model_df=order[0] + order[2],
        return_df=True
    )

    print("AIC:", fit.aic)
    print("BIC:", fit.bic)
    print("Ljung-Box:")
    print(lb)
    print("Hội tụ:", fit.mle_retvals.get("converged", None))
