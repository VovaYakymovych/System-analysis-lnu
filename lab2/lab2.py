import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Діапазон значень x1, x2
x1_values = np.arange(0, 2.01, 0.01)
x2_values = np.arange(0, 2.01, 0.01)

# Побудова сітки
X1, X2 = np.meshgrid(x1_values, x2_values)

# Цільові функції
F12 = 4 * X1**3 + 2 * X1 * X2 + X2**2 + 7
F21 = X1**3 + 3 * X1 * X2 - 2 * X2**2 - 8

# ====== 1. Гарантовані результати ======
# Табличний метод
f12_star = np.min(np.max(F12, axis=1))  # Мінімаксне значення для F12
f21_star = np.min(np.max(F21, axis=0))  # Мінімаксне значення для F21

print(f"Гарантований результат (табличний метод): f12* = {f12_star:.4f}, f21* = {f21_star:.4f}")

# Графічний метод: побудова теплових карт
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(F12, xticklabels=20, yticklabels=20, cmap="coolwarm", ax=ax[0])
ax[0].set_title("Графік f12(x1, x2)")
sns.heatmap(F21, xticklabels=20, yticklabels=20, cmap="coolwarm", ax=ax[1])
ax[1].set_title("Графік f21(x2, x1)")
plt.show()

# ====== 2. Пошук множини Парето ======
pareto_indices = (F12 >= f12_star) & (F21 >= f21_star)
pareto_x1, pareto_x2 = X1[pareto_indices], X2[pareto_indices]
pareto_f12, pareto_f21 = F12[pareto_indices], F21[pareto_indices]

# Відображення множини Парето
plt.scatter(pareto_f12, pareto_f21, color='red', label="Множина Парето")
plt.xlabel("f12(x1, x2)")
plt.ylabel("f21(x2, x1)")
plt.legend()
plt.title("Множина Парето")
plt.show()

# ====== 3. Оптимальні значення x1*, x2* ======
# Мінімізуємо ∆ = |f(x) - f*|
delta = np.abs(F12 - f12_star) + np.abs(F21 - f21_star)
min_index = np.unravel_index(np.argmin(delta), delta.shape)
x1_opt, x2_opt = x1_values[min_index[1]], x2_values[min_index[0]]

print(f"Оптимальні значення: x1* = {x1_opt:.2f}, x2* = {x2_opt:.2f}")
