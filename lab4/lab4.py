import numpy as np
import matplotlib.pyplot as plt

# === Чебишевські поліноми ===
def chebyshev_poly(n, x):
    x_mapped = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x_mapped
    else:
        T_prev, T_curr = np.ones_like(x), x_mapped
        for _ in range(2, n + 1):
            T_next = 2 * x_mapped * T_curr - T_prev
            T_prev, T_curr = T_curr, T_next
        return T_curr

# === Нормалізація ===
def normalize_vector(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def vector_norm(x):
    return np.sqrt(np.sum(x**2))

# === Побудова ознак через поліноми Чебишова ===
def build_feature_matrix(x1, x2, degree):
    n_samples = len(x1)
    features = [np.ones(n_samples)]
    for d1 in range(1, degree + 1):
        for d2 in range(1, degree + 1):
            features.append(chebyshev_poly(d1, x1) * chebyshev_poly(d2, x2))
    return np.vstack(features).T

# === Розв'язок задачі найменших квадратів ===
def least_squares_solution(X, y):
    return np.linalg.pinv(X) @ y

# === Основна функція моделі ===
def model_predict(X, coeffs):
    return X @ coeffs

# === Основна програма ===
# Зчитування даних
data = np.loadtxt('data.txt')
X_raw = data[:, :-5]  # перші 6 колонок - входи
Y_raw = data[:, -5:]  # останні 5 колонок - виходи

# Розділення входів на 3 групи по 2 змінні (по аналогії з x11,x12 ...)
X1 = normalize_vector(X_raw[:, 0:2])
X2 = normalize_vector(X_raw[:, 2:4])
X3 = normalize_vector(X_raw[:, 4:6])

# Тільки перший вихід y1 для спрощення
Y = normalize_vector(Y_raw[:, 0])

# Побудова ознак
degree = 3
Phi1 = build_feature_matrix(X1[:, 0], X1[:, 1], degree)
Phi2 = build_feature_matrix(X2[:, 0], X2[:, 1], degree)
Phi3 = build_feature_matrix(X3[:, 0], X3[:, 1], degree)

# Розрахунок коефіцієнтів
coeffs1 = least_squares_solution(Phi1, Y)
coeffs2 = least_squares_solution(Phi2, Y)
coeffs3 = least_squares_solution(Phi3, Y)

# Комбінування моделей (ваги однакові для простоти)
pred1 = model_predict(Phi1, coeffs1)
pred2 = model_predict(Phi2, coeffs2)
pred3 = model_predict(Phi3, coeffs3)
pred_final = (pred1 + pred2 + pred3) / 3

# Графік
plt.plot(Y, label='True')
plt.plot(pred_final, label='Predicted', linestyle='--')
plt.legend()
plt.title("Chebyshev Model Approximation")
plt.grid(True)
plt.show()