import numpy as np
import matplotlib.pyplot as plt


def f12(x1, x2):
    return 4 * x1 ** 3 + 2 * x1 * x2 + x2 ** 2 + 7


def f21(x1, x2):
    return x1 ** 3 + 3 * x1 * x2 - 2 * x2 ** 2 - 8


def grids(step=0.01, x_min=0, x_max=2):
    x_vals = np.arange(x_min, x_max + step, step)
    return x_vals, x_vals


def matrices(x1_vals, x2_vals):
    F12, F21 = np.meshgrid(x1_vals, x2_vals, indexing='ij')
    return f12(F12, F21), f21(F12, F21)


def result_1(F12, x1_vals, x2_vals):
    min_over_x2 = np.min(F12, axis=1)
    i_opt = np.argmax(min_over_x2)
    return min_over_x2[i_opt], x1_vals[i_opt], x2_vals[np.argmin(F12[i_opt, :])]


def result_2(F21, x1_vals, x2_vals):
    min_over_x1 = np.min(F21, axis=0)
    j_opt = np.argmax(min_over_x1)
    return min_over_x1[j_opt], x2_vals[j_opt], x1_vals[np.argmin(F21[:, j_opt])]


def pareto(F12, F21, x1_vals, x2_vals, f12_star, f21_star):
    mask = (F12 >= f12_star) & (F21 >= f21_star)
    x1_pareto, x2_pareto = np.meshgrid(x1_vals, x2_vals, indexing='ij')
    return list(zip(x1_pareto[mask], x2_pareto[mask], F12[mask], F21[mask]))


def find_optimal_point_by_deviations(F12, F21, x1_vals, x2_vals):
    f12_max, f21_max = np.max(F12), np.max(F21)
    deviations = np.abs(F12 - f12_max) + np.abs(F21 - f21_max)
    i_opt, j_opt = np.unravel_index(np.argmin(deviations), deviations.shape)
    return x1_vals[i_opt], x2_vals[j_opt], F12[i_opt, j_opt], F21[i_opt, j_opt]


def plot(F, x1_vals, x2_vals, title):
    X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing='ij')
    plt.figure(figsize=(6, 5))
    cp = plt.contourf(X1, X2, F, levels=30, cmap='jet')
    plt.colorbar(cp)
    plt.xlabel("x2")
    plt.ylabel("x1")
    plt.title(title)
    plt.grid(True)
    plt.show()


def main():
    step = 0.01
    x1_vals, x2_vals = grids(step)
    F12, F21 = matrices(x1_vals, x2_vals)

    f12_star, x1_opt_1, x2_min_1 = result_1(F12, x1_vals, x2_vals)
    print(f"Гарантований результат суб’єкта 1: f12* = {f12_star:.3f}, x1 = {x1_opt_1:.3f}, x2 = {x2_min_1:.3f}")

    f21_star, x2_opt_2, x1_min_2 = result_2(F21, x1_vals, x2_vals)
    print(f"Гарантований результат суб’єкта 2: f21* = {f21_star:.3f}, x2 = {x2_opt_2:.3f}, x1 = {x1_min_2:.3f}")

    pareto_points = pareto(F12, F21, x1_vals, x2_vals, f12_star, f21_star)
    print(f"Кількість точок Парето: {len(pareto_points)}")

    x1_opt_dev, x2_opt_dev, f12_opt_dev, f21_opt_dev = find_optimal_point_by_deviations(F12, F21, x1_vals, x2_vals)
    print(
        f"Точка, що мінімізує суму відхилень: x1* = {x1_opt_dev:.3f}, x2* = {x2_opt_dev:.3f}, f12 = {f12_opt_dev:.3f}, f21 = {f21_opt_dev:.3f}")

    plot(F12, x1_vals, x2_vals, "f12(x1, x2)")
    plot(F21, x1_vals, x2_vals, "f21(x1, x2)")


if __name__ == "__main__":
    main()
