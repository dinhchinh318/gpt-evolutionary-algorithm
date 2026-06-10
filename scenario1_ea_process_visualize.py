import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 0. Output folder
# =========================================================
OUTPUT_DIR = "ea_process_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "font.size": 10
})


# =========================================================
# 1. Benchmark functions (2D)
# =========================================================
def sphere(X):
    # X shape: (..., 2)
    return np.sum(X**2, axis=-1)

def rastrigin(X):
    x = X[..., 0]
    y = X[..., 1]
    return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))

def point_aiming(X, target=(1.0, 1.0)):
    tx, ty = target
    x = X[..., 0]
    y = X[..., 1]
    return np.sqrt((x - tx)**2 + (y - ty)**2)

def ackley(X):
    x = X[..., 0]
    y = X[..., 1]
    return (
        -20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2)))
        -np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
        + 20 + np.e
    )


# =========================================================
# 2. Simple GA for visualization
# =========================================================
def tournament_selection(pop, fitness, k=3):
    idx = np.random.choice(len(pop), size=k, replace=False)
    best = idx[np.argmin(fitness[idx])]
    return pop[best].copy()

def crossover(p1, p2):
    alpha = np.random.rand()
    child = alpha * p1 + (1 - alpha) * p2
    return child

def mutate(x, sigma=0.2):
    return x + np.random.normal(0, sigma, size=x.shape)

def run_simple_ga(
    f,
    bounds=(-5.12, 5.12),
    pop_size=40,
    generations=50,
    crossover_prob=0.8,
    mutation_sigma=0.25,
    seed=42
):
    np.random.seed(seed)

    lo, hi = bounds

    # initialize population
    pop = np.random.uniform(lo, hi, size=(pop_size, 2))
    fitness = f(pop)

    history_pops = [pop.copy()]
    history_best = [fitness.min()]
    history_best_point = [pop[np.argmin(fitness)].copy()]

    for _ in range(generations):
        offspring = []

        for _ in range(pop_size):
            if np.random.rand() < crossover_prob:
                p1 = tournament_selection(pop, fitness, k=3)
                p2 = tournament_selection(pop, fitness, k=3)
                child = crossover(p1, p2)
            else:
                p1 = tournament_selection(pop, fitness, k=3)
                child = p1.copy()

            child = mutate(child, sigma=mutation_sigma)
            child = np.clip(child, lo, hi)
            offspring.append(child)

        offspring = np.array(offspring)
        offspring_fitness = f(offspring)

        # elitist replacement: keep best pop_size among parents + offspring
        combined = np.vstack([pop, offspring])
        combined_fit = np.concatenate([fitness, offspring_fitness])

        best_idx = np.argsort(combined_fit)[:pop_size]
        pop = combined[best_idx]
        fitness = combined_fit[best_idx]

        history_pops.append(pop.copy())
        history_best.append(fitness.min())
        history_best_point.append(pop[np.argmin(fitness)].copy())

    return {
        "history_pops": history_pops,
        "history_best": history_best,
        "history_best_point": history_best_point,
        "final_pop": pop,
        "final_fitness": fitness,
        "best_x": pop[np.argmin(fitness)],
        "best_f": fitness.min()
    }


# =========================================================
# 3. Plot pipeline: Input -> EA -> Output
# =========================================================
def plot_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    boxes = [
        (0.08, 0.5, "INPUT\n\nFunction f(x)\nDimension p\nEA type"),
        (0.35, 0.5, "INITIALIZE\n\nGenerate random\ncandidate solutions"),
        (0.62, 0.5, "EVOLVE\n\nEvaluate fitness\nSelect better solutions\nCrossover / Mutation"),
        (0.88, 0.5, "OUTPUT\n\nBest solution x*\nBest fitness f(x*)")
    ]

    for x, y, text in boxes:
        ax.text(
            x, y, text,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#eaf2ff", edgecolor="black"),
            fontsize=11
        )

    arrowprops = dict(arrowstyle="->", linewidth=2, color="black")
    ax.annotate("", xy=(0.25, 0.5), xytext=(0.17, 0.5), arrowprops=arrowprops)
    ax.annotate("", xy=(0.52, 0.5), xytext=(0.44, 0.5), arrowprops=arrowprops)
    ax.annotate("", xy=(0.79, 0.5), xytext=(0.71, 0.5), arrowprops=arrowprops)

    ax.set_title("Experiment 1 / Scenario 1: Evolutionary Algorithm Process", fontsize=14, weight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "01_pipeline.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Saved", out)


# =========================================================
# 4. Contour helper
# =========================================================
def make_grid(bounds=(-5.12, 5.12), n=300):
    lo, hi = bounds
    x = np.linspace(lo, hi, n)
    y = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X, Y], axis=-1)
    return X, Y, XY

def plot_population_snapshots(f, func_name, result, bounds=(-5.12, 5.12), optimum=(0, 0)):
    X, Y, XY = make_grid(bounds=bounds, n=300)
    Z = f(XY)

    snapshots = [0, len(result["history_pops"]) // 2, len(result["history_pops"]) - 1]
    titles = ["Generation 0", "Middle Generation", "Final Generation"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, gen_idx, title in zip(axes, snapshots, titles):
        pop = result["history_pops"][gen_idx]
        best = result["history_best_point"][gen_idx]

        contour = ax.contourf(X, Y, Z, levels=40)
        ax.scatter(pop[:, 0], pop[:, 1], s=18, label="Population")
        ax.scatter(best[0], best[1], s=80, marker="*", label="Best in generation")
        ax.scatter(optimum[0], optimum[1], s=90, marker="X", label="Global optimum")

        ax.set_title(f"{title}\n(best fitness = {result['history_best'][gen_idx]:.4f})")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(loc="upper right", fontsize=8)

    fig.colorbar(contour, ax=axes, shrink=0.85, label="f(x)")
    fig.suptitle(f"Population Evolution on {func_name}", fontsize=15, weight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"02_population_snapshots_{func_name}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Saved", out)


# =========================================================
# 5. Convergence plot
# =========================================================
def plot_convergence(result, func_name):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result["history_best"], linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness so far")
    ax.set_title(f"Convergence Curve on {func_name}")
    ax.grid(True, linestyle=":", alpha=0.5)

    out = os.path.join(OUTPUT_DIR, f"03_convergence_{func_name}.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Saved", out)


# =========================================================
# 6. Multiple runs boxplot
# =========================================================
def plot_boxplot_multiple_runs(f, func_name, bounds=(-5.12, 5.12), runs=20):
    final_best = []
    for seed in range(runs):
        result = run_simple_ga(
            f=f,
            bounds=bounds,
            pop_size=40,
            generations=50,
            crossover_prob=0.8,
            mutation_sigma=0.25,
            seed=seed
        )
        final_best.append(result["best_f"])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.boxplot([final_best], tick_labels=[func_name], showfliers=False)
    ax.set_ylabel("Final best fitness")
    ax.set_title(f"Distribution of Final Best Fitness\n({runs} runs)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    out = os.path.join(OUTPUT_DIR, f"04_boxplot_{func_name}.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Saved", out)


# =========================================================
# 7. Landscape gallery
# =========================================================
def plot_landscape_gallery():
    functions = [
        ("Sphere", sphere, (-5.12, 5.12), (0, 0)),
        ("Point_Aiming", lambda X: point_aiming(X, target=(1, 1)), (-3, 4), (1, 1)),
        ("Ackley", ackley, (-5, 5), (0, 0)),
        ("Rastrigin", rastrigin, (-5.12, 5.12), (0, 0)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()

    for ax, (name, f, bounds, opt) in zip(axes, functions):
        X, Y, XY = make_grid(bounds=bounds, n=250)
        Z = f(XY)
        contour = ax.contourf(X, Y, Z, levels=40)
        ax.scatter(opt[0], opt[1], marker="X", s=100)
        ax.set_title(name)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    fig.colorbar(contour, ax=axes, shrink=0.85, label="f(x)")
    fig.suptitle("2D Landscapes of Synthetic Problems", fontsize=15, weight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "05_landscape_gallery.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Saved", out)


# =========================================================
# 8. Main
# =========================================================
def main():
    print("Creating EA process visualizations...")

    # 1) pipeline
    plot_pipeline_diagram()

    # 2) landscape gallery
    plot_landscape_gallery()

    # 3) choose one function to show EA process clearly
    func_name = "Rastrigin"
    f = rastrigin
    bounds = (-5.12, 5.12)
    optimum = (0, 0)

    result = run_simple_ga(
        f=f,
        bounds=bounds,
        pop_size=40,
        generations=50,
        crossover_prob=0.8,
        mutation_sigma=0.25,
        seed=42
    )

    plot_population_snapshots(f, func_name, result, bounds=bounds, optimum=optimum)
    plot_convergence(result, func_name)
    plot_boxplot_multiple_runs(f, func_name, bounds=bounds, runs=20)

    print("\nDone.")
    print(f"Images saved in folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()