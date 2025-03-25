import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Завантаження даних
path = "132-serednia-nominalna-zarobitna-plata-za-misiats-grn.csv"
df = pd.read_csv(path, delimiter=";", encoding="1251")

# Підготовка даних
rates = df[["period", "data"]].copy()
rates["period"] = pd.to_datetime(rates["period"], dayfirst=True)
rates["data"] = pd.to_numeric(rates["data"], errors="coerce")
rates = rates.sort_values("period").dropna()

# Агрегація
aggregated_df = rates.groupby("period", as_index=False).agg({"data": "sum"})
measurements = aggregated_df["data"].values
dates = aggregated_df["period"].values

# Параметри для фільтра Калмана
dt = 1  # Крок часу
n = len(aggregated_df)
A = np.array([[1, dt], [0, 1]])  # Матриця переходу стану
H = np.array([[1, 0]])  # Матриця вимірювання
Q = np.array([[1e-4, 0], [0, 1e-4]])  # Коваріація шуму процесу
R = np.array([[10]])  # Коваріація шуму вимірювання
I = np.eye(2)  # Одинична матриця 2x2

# Ініціалізація Калмана
x = np.array([[measurements[0]], [0]])  # Початковий стан: [y0, 0]
P = np.eye(2) * 1000  # Початкова коваріація

# Масиви для зберігання результатів
x_estimates = []  # Калман
poly_estimates = []  # Поліном
dates_processed = []
poly_coeffs_history = []

# Емуляція реального часу
print("Емуляція реального часу: обробка вимірювань по черзі")
for k in range(n):
    # Імітуємо нове вимірювання
    z = np.array([[measurements[k]]])
    current_date = dates[k]

    # --- Фільтр Калмана ---
    x_pred = A.dot(x)
    P_pred = A.dot(P).dot(A.T) + Q
    innovation = z - H.dot(x_pred)
    S = H.dot(P_pred).dot(H.T) + R
    K = P_pred.dot(H.T).dot(np.linalg.inv(S))
    x = x_pred + K.dot(innovation)
    P = (I - K.dot(H)).dot(P_pred)
    x_estimates.append(x[0, 0])

    # --- Квадратичний поліном ---
    x_poly = np.arange(k + 1)
    y_poly = measurements[:k + 1]
    if k < 2:
        coeffs = np.polyfit(x_poly, y_poly, min(k, 1))
        if k == 0:
            coeffs = np.array([measurements[0]])
        else:
            coeffs = np.pad(coeffs, (2 - len(coeffs), 0), mode='constant')
    else:
        coeffs = np.polyfit(x_poly, y_poly, 2)
    poly_coeffs_history.append(coeffs)
    poly_val = np.polyval(coeffs, x_poly[-1])
    poly_estimates.append(poly_val)

    # Збереження дати
    dates_processed.append(current_date)

    # Виведення
    print(f"Крок {k + 1}/{n}: Дата = {current_date}, Вимірювання = {z[0, 0]:.2f}, "
          f"Калман = {x[0, 0]:.2f}, Поліном = {poly_val:.2f}")

# Прогнозування на половину діапазону
days_range = int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))  # Перетворюємо timedelta64 у дні
forecast_days = days_range // 2
forecast_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')

# --- Прогноз Калмана ---
forecast_x = np.zeros((forecast_days, 2))
forecast_x[0] = x.flatten()
for k in range(1, forecast_days):
    forecast_x[k] = A.dot(forecast_x[k - 1])

# --- Прогноз полінома ---
forecast_x_poly = np.arange(n, n + forecast_days)
last_poly_coeffs = poly_coeffs_history[-1]
forecast_poly = np.polyval(last_poly_coeffs, forecast_x_poly)

# Підготовка до візуалізації
df_plot = pd.DataFrame({"period": dates_processed, "data": measurements[:len(dates_processed)],
                        "Kalman": x_estimates, "Poly": poly_estimates})
forecast_df = pd.DataFrame({"period": forecast_dates, "Kalman": forecast_x[:, 0], "Poly": forecast_poly})

# Візуалізація
plt.figure(figsize=(14, 7))
plt.plot(df_plot["period"], df_plot["data"], label="Реальні дані", color='blue', alpha=0.5)
plt.plot(df_plot["period"], df_plot["Kalman"], label="Фільтр Калмана", color='orange', linestyle='--')
plt.plot(df_plot["period"], df_plot["Poly"], label="Квадратичний поліном", color='purple', linestyle='--')
plt.plot(forecast_df["period"], forecast_df["Kalman"], label="Прогноз Калмана", color='green', linestyle='-.')
plt.plot(forecast_df["period"], forecast_df["Poly"], label="Прогноз полінома", color='red', linestyle='-.')
plt.legend()
plt.xlabel("Дата")
plt.ylabel("Середня заробітна плата")
plt.title("Порівняння Калмана та квадратичного полінома (реальний час)")
plt.grid()
plt.show()

# Порівняння точності
errors_kalman = measurements[:len(x_estimates)] - np.array(x_estimates)
errors_poly = measurements[:len(poly_estimates)] - np.array(poly_estimates)
mse_kalman = np.mean(errors_kalman ** 2)
mse_poly = np.mean(errors_poly ** 2)
print(f"Середньоквадратична помилка (MSE) Калмана: {mse_kalman:.2f}")
print(f"Середньоквадратична помилка (MSE) Полінома: {mse_poly:.2f}")

# Гістограма помилок
plt.figure(figsize=(10, 5))
plt.hist(errors_kalman, bins=20, alpha=0.6, color='orange', label="Помилки Калмана", density=True)
plt.hist(errors_poly, bins=20, alpha=0.6, color='purple', label="Помилки полінома", density=True)
plt.title("Гістограма помилок Калмана та полінома")
plt.xlabel("Помилка")
plt.ylabel("Щільність")
plt.legend()
plt.grid()
plt.show()