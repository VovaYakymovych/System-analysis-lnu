import numpy as np
import matplotlib.pyplot as plt

# Визначаємо діапазон та крок
x_values = np.arange(6, 8.001, 0.001)

# Функції
f1 = 15 * np.sin(x_values + 1)
f2 = 10 * np.cos(2 * x_values - 2.4) + 12

# Обмеження
f1_star = 12.82
f2_star = 16

# Відбір допустимих рішень
valid_indices = (f1 <= f1_star) & (f2 >= f2_star)
valid_x = x_values[valid_indices]
valid_f1 = f1[valid_indices]
valid_f2 = f2[valid_indices]

# Побудова множини Парето
pareto_indices = []
for i in range(len(valid_x)):
    dominated = False
    for j in range(len(valid_x)):
        if (valid_f1[j] <= valid_f1[i] and valid_f2[j] >= valid_f2[i]) and (valid_f1[j] < valid_f1[i] or valid_f2[j] > valid_f2[i]):
            dominated = True
            break
    if not dominated:
        pareto_indices.append(i)

pareto_x = valid_x[pareto_indices]
pareto_f1 = valid_f1[pareto_indices]
pareto_f2 = valid_f2[pareto_indices]


plt.figure(figsize=(8, 6))
plt.scatter(valid_f1, valid_f2, label="Допустимі рішення", color="gray", alpha=0.5)
plt.scatter(pareto_f1, pareto_f2, label="Множина Парето", color="red")
plt.xlabel("f1(x)")
plt.ylabel("f2(x)")
plt.legend()
plt.title("Множина Парето")
plt.show()


print("Множина Парето (округлено до 0.001):")
for x, f1_val, f2_val in zip(pareto_x, pareto_f1, pareto_f2):
    print(f"x = {round(x, 3)}, f1 = {round(f1_val, 3)}, f2 = {round(f2_val, 3)}")
