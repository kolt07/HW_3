import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Отримання даних у вигляді ДФ
def get_data():
    path = "132-serednia-nominalna-zarobitna-plata-za-misiats-grn.csv"
    df = pd.read_csv(path, encoding="1251", delimiter=";")
    return df

# Очищення, агрегація даних
def prepare_data(df):
    rates = df[["period", "data"]].copy()
    rates["period"] = pd.to_datetime(rates["period"], dayfirst=True)
    rates["data"] = pd.to_numeric(rates["data"], errors="coerce")
    rates = rates.sort_values("period").dropna()
    aggregated_df = rates.groupby("period", as_index=False).agg({"data": "sum"})
    return aggregated_df

# Отримання тренду заданого ступеню
def trend(degree, x, y):
    coef = np.polyfit(x, y, degree)
    trend = np.polyval(coef, x)
    return trend

# Обчислення р-квадрату для тренду
def r_square(y, trend):
    residuals_initial = y - trend
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(residuals_initial ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# Отримання даних фільтрованих з емуляцією покрокового надходження
def ABF(S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    YoutAB = np.zeros((iter, 1))
    T0 = 1
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
    Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / T0
    Yextra = Yin[0, 0] + Yspeed_retro
    alfa = 1 * (1 * 1 - 1) / (1 * (1 + 1))
    beta = (3 / 1) * (1 + 1)
    YoutAB[0, 0] = Yin[0, 0] + alfa * (Yin[0, 0])
    for i in range(1, iter):
        YoutAB[i, 0] = Yextra + alfa * (Yin[i, 0] - Yextra)
        Yspeed = Yspeed_retro + (beta / T0) * (Yin[i, 0] - Yextra)
        Yspeed_retro = Yspeed
        Yextra = YoutAB[i, 0] + Yspeed_retro
        alfa = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 / (i * (i + 1))
    return YoutAB


def kalman_filter(S0):
    iter = len(S0)
    YoutK = np.zeros(iter)  # Вихідний масив для відфільтрованих значень

    # Ініціалізація першим значенням
    x_est = float(S0[0])  # Початковий стан = перше вимірювання
    v_est = 0.0  # Початкова швидкість (тренд)
    P = np.array([[1.0, 0.0], [0.0, 1.0]])  # Початкова коваріація помилки (2x2 матриця)

    # Параметри
    dt = 1.0  # Крок часу
    Q = np.array([[1e-4, 0.0], [0.0, 1e-4]])  # Шум процесу
    R = 10.0  # Шум вимірювань
    F = np.array([[1.0, dt], [0.0, 1.0]])  # Матриця переходу стану
    H = np.array([1.0, 0.0])  # Матриця спостереження

    for i in range(iter):
        z = float(S0[i])  # Вимірювання на поточному кроці

        # Крок передбачення
        x_pred = F @ np.array([x_est, v_est])  # Передбачення стану
        P_pred = F @ P @ F.T + Q  # Передбачення коваріації

        # Крок оновлення
        innovation = z - H @ x_pred  # Інновація (помилка вимірювання)
        S = H @ P_pred @ H.T + R  # Коваріація інновації
        K = (P_pred @ H.T) / S  # Калманівське підсилення
        K = K.ravel()  # Забезпечуємо, що K є одновимірним

        # Оновлення стану
        state_update = x_pred + K * innovation  # (2,) + (2,) * scalar = (2,)
        x_est = state_update[0]  # Скаляр
        v_est = state_update[1]  # Скаляр
        P = P_pred - np.outer(K, H) @ P_pred  # Оновлення коваріації

        YoutK[i] = x_est  # Зберігаємо відфільтроване значення

    return YoutK

def get_data_for_analysys(prepared_data):
    x = (prepared_data["period"] - prepared_data["period"].min()).dt.days
    y = prepared_data["data"].to_numpy()
    return x, y

def cumulative_absolute_error(y_true, y_pred):
    absolute_errors = np.abs(y_true - y_pred)
    cumulative_errors = np.cumsum(absolute_errors)
    return cumulative_errors

# Підготовка даних
raw_data = get_data()
prepared_data = prepare_data(raw_data)
x, y = get_data_for_analysys(prepared_data)
data_trend = trend(3, x, y)
filtered_data_abf = ABF(y).ravel()  # Використовую ravel() замість flatten()
filtered_data_kalman = kalman_filter(y)

# Обчислення R²
rsq_trend = r_square(y, data_trend)
rsq_ABF = r_square(y, filtered_data_abf)
rsq_kalman = r_square(y, filtered_data_kalman)

print(f"R² (тренд за фінальними даними): {rsq_trend}")
print(f"R² (тренд за АБФ даними): {rsq_ABF}")
print(f"R² (тренд за даними фільтрації Калмана): {rsq_kalman}")

# Обчислення накопиченої похибки
cae_trend = cumulative_absolute_error(y, data_trend)
cae_abf = cumulative_absolute_error(y, filtered_data_abf)
cae_kalman = cumulative_absolute_error(y, filtered_data_kalman)

# Загальна накопичена похибка (кінцеве значення)
total_cae_trend = cae_trend[-1]
total_cae_abf = cae_abf[-1]
total_cae_kalman = cae_kalman[-1]

print(f"Загальна накопичена похибка (тренд): {total_cae_trend}")
print(f"Загальна накопичена похибка (АБФ): {total_cae_abf}")
print(f"Загальна накопичена похибка (Калман): {total_cae_kalman}")

# Графік 1: Реальні дані та моделі
plt.figure(figsize=(12, 6))
plt.plot(prepared_data["period"], y, label="Реальні дані", color="black")
plt.plot(prepared_data["period"], data_trend, label="Тренд (ступінь 3)", color="blue")
plt.plot(prepared_data["period"], filtered_data_abf, label="АБФ фільтр", color="red")
plt.plot(prepared_data["period"], filtered_data_kalman, label="Калман фільтр", color="green")
plt.xlabel("Період")
plt.ylabel("Значення")
plt.title("Реальні дані та моделі")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Графік 2: Накопичена похибка з часом
plt.figure(figsize=(12, 6))
plt.plot(prepared_data["period"], cae_trend, label="Накопичена похибка (тренд)", color="blue")
plt.plot(prepared_data["period"], cae_abf, label="Накопичена похибка (АБФ)", color="red")
plt.plot(prepared_data["period"], cae_kalman, label="Накопичена похибка (Калман)", color="green")
plt.xlabel("Період")
plt.ylabel("Накопичена похибка")
plt.title("Зміна накопиченої похибки з часом")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()