import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt

# Сітка значень
x_range = np.arange(0, 6)
y_range = np.arange(0, 6)


# Функції прибутку
def profit_A(x, y):
    return -1.2 * x ** 2 + 11 * x + y ** 2 - 9 * y + 2.1


def profit_B(x, y):
    return x ** 2 - 4.5 * x + 2.8 * y - 1.7 * y ** 2 + 0.2 * x * y + 28


# ----- Витрати для сторони A -----

def cost_A_method1(x, y):
    # Простий метод: витрати лінійно залежать тільки від y (наприклад, ринкові умови)
    return 0.12 * (0.1 * y + 2.3)

def cost_A_method2(x, y):
    # Нелінійний метод: враховує взаємодію x і y, масштаб, та корекцію по y
    return (0.07 * x * y - 0.15 * y + 1.9 * x + 3) / 2.8

def cost_A_method3(x, y):
    # Змішаний метод: лінійна залежність від x, y, і їх взаємодії з базовим зсувом
    return (y - 0.08 * x * y + 1.5 * x + 18) / 9.5

# ----- Витрати для сторони B -----

def cost_B_method1(x, y):
    # Множинна модель: витрати залежать від добутку y та (лінійної функції x)
    return 0.25 * y * (0.08 * x + 2.8)

def cost_B_method2(x, y):
    # Комбінована модель: включає прямі та змішані терміни з базовим рівнем
    return (y + 0.03 * x * y - 0.3 * x + 3.5) / 1.9

def cost_B_method3(x, y):
    # Псевдо-лінійна модель: залежність від y, x і x*y з невеликою поправкою
    return (y - 0.12 * x * y - 0.3 + x) / 2.8

# Ваги методів залежно від координат
weights_A = np.array([
    [0.15, 0.1, 0.2, 0.3, 0.25, 0.2],
    [0.04, 0.06, 0.05, 0.08, 0.07, 0.06],
    [0.11, 0.18, 0.28, 0.2, 0.22, 0.16]
])

weights_B = np.array([
    [0.1, 0.15, 0.25, 0.35, 0.22, 0.1],
    [0.06, 0.12, 0.14, 0.17, 0.2, 0.18],
    [0.32, 0.27, 0.29, 0.38, 0.42, 0.36]
])

# Результати
outcomes = []

for x, y in product(x_range, y_range):
    pA = profit_A(x, y)
    pB = profit_B(x, y)

    wA1, wA2, wA3 = weights_A[:, y]
    wB1, wB2, wB3 = weights_B[:, x]

    cA = (
            cost_A_method1(x, y) * wA1 +
            cost_A_method2(x, y) * wA2 +
            cost_A_method3(x, y) * wA3
    )

    cB = (
            cost_B_method1(x, y) * wB1 +
            cost_B_method2(x, y) * wB2 +
            cost_B_method3(x, y) * wB3
    )

    total_profit = pA + pB
    total_cost = cA + cB
    result = total_profit - total_cost

    outcomes.append((x, y, total_profit, total_cost, result))

# Таблиця
df = pd.DataFrame(outcomes, columns=["x", "y", "Profit", "Cost", "NetResult"])
print(df)

# Оптимальне рішення
best = df.loc[df["NetResult"].idxmax()]
print(f"\nОптимальне рішення: x = {best['x']}, y = {best['y']}, NetResult = {best['NetResult']:.3f}")

# Створимо матрицю значень NetResult
Z = np.zeros((len(y_range), len(x_range)))
for row in outcomes:
    x, y, _, _, res = row
    Z[y, x] = res  # важливо: y — рядки, x — стовпці

# Побудова графіка
plt.figure(figsize=(8, 6))
cp = plt.contourf(x_range, y_range, Z, levels=20, cmap="viridis")
plt.colorbar(cp, label="NetResult")
plt.title("Контурна карта NetResult")
plt.xlabel("x")
plt.ylabel("y")

# Позначення оптимального рішення
plt.plot(best["x"], best["y"], "ro", label="Оптимум")
plt.legend()
plt.grid(True)
plt.show()