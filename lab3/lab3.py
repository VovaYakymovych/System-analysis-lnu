import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg

# 1. Генерація випадкових даних
np.random.seed(42)

# Створення назв ознак та виходів
feature_cols = [f'X{i}{j}' for i in range(1, 4) for j in range(1, 3)]  # X11, X12, ..., X32
target_cols = [f'Y{i}' for i in range(1, 6)]  # Y1, ..., Y5

# Діапазони для ознак: верхні межі випадкові
feature_ranges = {col: (0, np.random.uniform(3, 4.5)) for col in feature_cols}
features = pd.DataFrame({
    col: np.around(np.random.uniform(low, high, 40), 2)
    for col, (low, high) in feature_ranges.items()
})

# Залежності вихідних даних від ознак
target_weights = {
    'Y1': {'X11': 2.1, 'X12': 1.6},
    'Y2': {'X21': 1.3, 'X22': 1.9},
    'Y3': {'X31': 1.4, 'X32': 1.2},
    'Y4': {'X11': 2.0, 'X21': 1.5},
    'Y5': {'X12': 1.7, 'X32': 1.5}
}

# Формування вихідних параметрів з шумом
targets = pd.DataFrame({
    y_col: np.around(
        sum(features[x_col] * weight for x_col, weight in weights.items()) +
        np.random.uniform(-1.0, 1.0, 40), 2
    )
    for y_col, weights in target_weights.items()
})

# Об'єднання в один DataFrame
data = pd.concat([features, targets], axis=1)

# 2. Нормалізація виходів
X_matrix = data[feature_cols].to_numpy()
Y_matrix = data[target_cols].to_numpy()
Y_min, Y_max = np.min(Y_matrix, axis=0), np.max(Y_matrix, axis=0)
normalized_Y = (Y_matrix - Y_min) / (Y_max - Y_min)

# 3. Метод спряжених напрямків
A_matrix = np.dot(X_matrix.T, X_matrix)

coefficients = {}
predicted_Y = np.zeros_like(normalized_Y)

for idx, target in enumerate(target_cols):
    b_vector = np.dot(X_matrix.T, normalized_Y[:, idx])
    sol, status = cg(A_matrix, b_vector)

    if status == 0:
        print(f"✅ Розв'язок для {target} знайдено.")
    elif status > 0:
        print(f"⚠️  Немає збіжності за {status} ітерацій для {target}.")
    else:
        print(f"❌ Помилка під час обчислення {target}.")

    coefficients[target] = sol
    predicted_Y[:, idx] = np.dot(X_matrix, sol)

# Вивід коефіцієнтів
coef_df = pd.DataFrame(coefficients, index=feature_cols)
print("\nКоефіцієнти моделі:")
print(coef_df)

# 4. Візуалізація результатів для Y1
plt.figure(figsize=(10, 5))
plt.plot(normalized_Y[:, 0], label='Actual', linestyle='-')
plt.plot(predicted_Y[:, 0], 'o', label='Predicted', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Normalized Value')
plt.title(f'Actual vs Predicted for {target_cols[0]}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
