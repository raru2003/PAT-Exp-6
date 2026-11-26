# Experiment 6: ARIMA Model for Forecasting

## Aim

To evaluate predictive models using performance metrics.

## Algorithm

1.  **START**
2.  Prepare stationary time series.
3.  Identify ARIMA -- (p, d, q) parameters using ACF/PACF.
4.  Fit the ARIMA model on the data.
5.  Forecast future periods and plot the results.
6.  Evaluate forecast accuracy.
7.  **STOP**



## Program
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = [100, 120, 130, 125, 140, 150, 160, 155, 170, 180]

train = data[:-2]
test = data[-2:]

model = ARIMA(train, order=(1, 1, 1))
fit = model.fit()

forecast = fit.forecast(steps=2)

print("Forecast:", forecast)
print("MAE:", mean_absolute_error(test, forecast))
print("MSE:", mean_squared_error(test, forecast))

plt.plot(data, label="Actual")
plt.plot(range(8, 10), forecast, label="Forecast")
plt.legend()
plt.title("ARIMA Forecasting")
plt.show()


## Output

Forecast: [approximately 169, 175]

MAE: Approximately 3
MSE: Approximately 11
<img width="725" height="375" alt="image" src="https://github.com/user-attachments/assets/2fafbab2-af82-4c97-b80e-7f6a4d0deaea" />

## Result
