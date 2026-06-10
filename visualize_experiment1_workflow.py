"""
visualize_experiment1_workflow.py
=================================
Tao anh truc quan hoa quy trinh thiet ke Thuc nghiem 1 / Scenario 1.

Noi dung anh dau ra:
  01_workflow_overview.png          : Tong quan input -> EA -> output
  02_experiment_design.png          : Ma tran thiet ke thuc nghiem
  03_ea_loop_detail.png             : Vong lap EA tu sinh nghiem
  04_candidate_to_fitness.png       : Vector nghiem x -> ham f(x) -> fitness
  05_landscape_gallery.png          : Landscape 2D cua cac ham synthetic
  06_population_evolution.png       : Population di chuyen qua cac generation
  07_convergence_ga_de.png          : Duong hoi tu cua GA va DE
  08_output_artifacts.png           : Output cuoi cung va cach dung cho bao cao

Cach chay:
  python visualize_experiment1_workflow.py

Thu vien can co:
  pip install numpy matplotlib

Ghi chu:
  - Script nay khong can GPU.
  - Tat ca hinh luu trong thu muc: exp1_workflow_visuals/
  - Code tap trung vao truc quan hoa quy trinh; khong can chay lai toan bo JGEA.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# ============================================================
# 0. Global config
# ============================================================
OUT_DIR = "exp1_workflow_visuals"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

COLORS = {
    "blue": "#2F6BFF",
    "dark_blue": "#183A8C",
    "light_blue": "#EAF1FF",
    "green": "#22A06B",
    "light_green": "#EAFBF0",
    "orange": "#F59E0B",
    "light_orange": "#FFF6E5",
    "red": "#E5484D",
    "light_red": "#FDECEC",
    "purple": "#7C3AED",
    "light_purple": "#F3E8FF",
    "gray": "#4B5563",
    "light_gray": "#F3F4F6",
    "black": "#111827",
}

# ============================================================
# 1. Utility drawing functions
# ============================================================

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def add_box(ax, xy, w, h, title, body="", fc="#FFFFFF", ec="#111827", title_color="#111827"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.3,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.08, title, ha="center", va="top",
            fontsize=12, weight="bold", color=title_color)
    if body:
        ax.text(x + w/2, y + h/2 - 0.03, body, ha="center", va="center",
                fontsize=9.5, color=COLORS["black"], linespacing=1.35)
    return box


def add_arrow(ax, start, end, color="#111827", lw=2.2, text=None, text_offset=(0, 0)):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                            linewidth=lw, color=color, shrinkA=6, shrinkB=6)
    ax.add_patch(arrow)
    if text:
        tx = (start[0] + end[0]) / 2 + text_offset[0]
        ty = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(tx, ty, text, ha="center", va="center", fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

# ============================================================
# 2. Benchmark functions for visualization
# ============================================================

def sphere(X):
    return np.sum(X ** 2, axis=-1)


def point_aiming(X, target_value=1.0):
    target = np.full(X.shape[-1], target_value)
    return np.linalg.norm(X - target, axis=-1)


def ackley(X, a=20.0, b=0.2, c=2*np.pi):
    n = X.shape[-1]
    sum_sq = np.sum(X ** 2, axis=-1)
    sum_cos = np.sum(np.cos(c * X), axis=-1)
    return -a*np.exp(-b*np.sqrt(sum_sq/n)) - np.exp(sum_cos/n) + a + np.e


def rastrigin(X, A=10.0):
    n = X.shape[-1]
    return A*n + np.sum(X**2 - A*np.cos(2*np.pi*X), axis=-1)


def griewank(X):
    n = X.shape[-1]
    idx = np.arange(1, n + 1)
    return np.sum(X**2, axis=-1) / 4000.0 - np.prod(np.cos(X / np.sqrt(idx)), axis=-1) + 1.0


def rosenbrock(X):
    return np.sum(100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (1 - X[..., :-1]) ** 2, axis=-1)


def cpa_targets_2d():
    # Project/paper-style CPA: 5 target points on circumference centered at (1, 1), radius 2.
    center = np.array([1.0, 1.0])
    radius = 2.0
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    return np.array([center + radius*np.array([np.cos(a), np.sin(a)]) for a in angles])


CPA_T = cpa_targets_2d()


def cpa_2d(X):
    dists = []
    for t in CPA_T:
        dists.append(np.linalg.norm(X - t, axis=-1))
    return np.min(np.stack(dists, axis=-1), axis=-1)


def make_grid(bounds, n=260):
    lo, hi = bounds
    xs = np.linspace(lo, hi, n)
    ys = np.linspace(lo, hi, n)
    X1, X2 = np.meshgrid(xs, ys)
    X = np.stack([X1, X2], axis=-1)
    return X1, X2, X

# ============================================================
# 3. Simple GA and DE for process visualization
# ============================================================

def tournament(pop, fit, k=3):
    ids = np.random.choice(len(pop), size=k, replace=False)
    return pop[ids[np.argmin(fit[ids])]].copy()


def run_ga_2d(fn, bounds=(-5.12, 5.12), pop_size=60, generations=60,
              xo_prob=0.8, mut_sigma=0.24, cross_sigma=0.04, seed=7):
    np.random.seed(seed)
    lo, hi = bounds
    pop = np.random.uniform(lo, hi, size=(pop_size, 2))
    fit = fn(pop)
    hist_pop = [pop.copy()]
    hist_best = [float(np.min(fit))]
    hist_best_point = [pop[np.argmin(fit)].copy()]

    for _ in range(generations):
        children = []
        for _ in range(pop_size):
            if np.random.rand() < xo_prob:
                p1 = tournament(pop, fit, k=3)
                p2 = tournament(pop, fit, k=3)
                alpha = np.random.rand()
                child = p1 + alpha * (p2 - p1) + np.random.normal(0, cross_sigma, 2)
            else:
                p = tournament(pop, fit, k=3)
                child = p + np.random.normal(0, mut_sigma, 2)
            children.append(np.clip(child, lo, hi))

        children = np.array(children)
        cfit = fn(children)
        all_pop = np.vstack([pop, children])
        all_fit = np.concatenate([fit, cfit])
        keep = np.argsort(all_fit)[:pop_size]
        pop = all_pop[keep]
        fit = all_fit[keep]
        hist_pop.append(pop.copy())
        hist_best.append(float(np.min(fit)))
        hist_best_point.append(pop[np.argmin(fit)].copy())

    return {
        "hist_pop": hist_pop,
        "hist_best": hist_best,
        "hist_best_point": hist_best_point,
        "best_x": pop[np.argmin(fit)],
        "best_f": float(np.min(fit)),
    }


def run_de_2d(fn, bounds=(-5.12, 5.12), pop_size=40, generations=60, F=0.5, CR=0.8, seed=11):
    np.random.seed(seed)
    lo, hi = bounds
    pop = np.random.uniform(lo, hi, size=(pop_size, 2))
    fit = fn(pop)
    hist_best = [float(np.min(fit))]

    for _ in range(generations):
        for i in range(pop_size):
            choices = [j for j in range(pop_size) if j != i]
            a, b, c = pop[np.random.choice(choices, 3, replace=False)]
            mutant = a + F * (b - c)
            trial = pop[i].copy()
            forced = np.random.randint(2)
            for j in range(2):
                if np.random.rand() < CR or j == forced:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lo, hi)
            tfit = fn(trial.reshape(1, 2))[0]
            if tfit < fit[i]:
                pop[i] = trial
                fit[i] = tfit
        hist_best.append(float(np.min(fit)))

    return {"hist_best": hist_best, "best_x": pop[np.argmin(fit)], "best_f": float(np.min(fit))}

# ============================================================
# 4. Figures
# ============================================================

def fig_01_workflow_overview():
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Experiment 1 Workflow: Synthetic Problem Optimization",
            ha="center", va="top", fontsize=18, weight="bold", color=COLORS["black"])
    ax.text(0.5, 0.895, "EA tự sinh nghiệm, đánh giá bằng f(x), chọn nghiệm tốt, tạo thế hệ mới, rồi trả về nghiệm tốt nhất.",
            ha="center", va="top", fontsize=10.5, color=COLORS["gray"])

    add_box(ax, (0.035, 0.45), 0.19, 0.28, "INPUT",
            "Problem name\nSphere / PA / CPA / ...\n\nDimension p\n20 / 100 / 200 / 500\n\nSolver\nGA / DE / ES / PSO / CMA-ES",
            fc=COLORS["light_blue"], ec=COLORS["blue"])

    add_box(ax, (0.285, 0.45), 0.18, 0.28, "CANDIDATES",
            "EA sinh population\n\nMỗi nghiệm là vector:\nx = [x1, x2, ..., xp]\n\nBan đầu thường random\nquanh [-1, 1]^p",
            fc=COLORS["light_green"], ec=COLORS["green"])

    add_box(ax, (0.525, 0.45), 0.19, 0.28, "FITNESS EVALUATION",
            "Đưa từng vector x\nvào hàm mục tiêu\n\nfitness = f(x)\n\nMinimization:\nfitness càng nhỏ càng tốt",
            fc=COLORS["light_orange"], ec=COLORS["orange"])

    add_box(ax, (0.765, 0.45), 0.20, 0.28, "EA UPDATE",
            "Selection\nMutation\nCrossover / recombination\n\nSinh nghiệm mới\nrồi đánh giá tiếp\n\nDừng ở budget evals",
            fc=COLORS["light_purple"], ec=COLORS["purple"])

    add_box(ax, (0.38, 0.08), 0.24, 0.22, "OUTPUT OF ONE RUN",
            "best_x\nbest_fitness = f(best_x)\nconvergence history",
            fc=COLORS["light_red"], ec=COLORS["red"])

    add_box(ax, (0.68, 0.08), 0.24, 0.22, "OUTPUT AFTER MANY RUNS",
            "final fitness distribution\nNER / EtTQ / NoVS\nfigures for report and slides",
            fc="#FFF7ED", ec="#EA580C")

    add_arrow(ax, (0.225, 0.59), (0.285, 0.59), color=COLORS["blue"])
    add_arrow(ax, (0.465, 0.59), (0.525, 0.59), color=COLORS["green"])
    add_arrow(ax, (0.715, 0.59), (0.765, 0.59), color=COLORS["orange"])
    add_arrow(ax, (0.86, 0.45), (0.62, 0.38), color=COLORS["purple"], text="repeat", text_offset=(0.02, 0.03))
    add_arrow(ax, (0.62, 0.45), (0.50, 0.30), color=COLORS["red"])
    add_arrow(ax, (0.62, 0.19), (0.68, 0.19), color="#EA580C")

    # device note
    ax.text(0.035, 0.035,
            "Thiết bị của bạn: Intel i7-12700H, 32 GB RAM, RTX 3050 Laptop GPU. Thực nghiệm synthetic chủ yếu dùng CPU; GPU không bắt buộc.",
            ha="left", va="bottom", fontsize=9.3, color=COLORS["gray"])

    savefig("01_workflow_overview.png")


def fig_02_experiment_design():
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Experiment Design Matrix", ha="center", va="top",
            fontsize=18, weight="bold")
    ax.text(0.5, 0.915, "Mỗi ô tương ứng một problem instance; mỗi instance được chạy với nhiều EA và nhiều seed.",
            ha="center", va="top", fontsize=10.2, color=COLORS["gray"])

    functions = ["Sphere", "PA-1", "PA-10", "CPA", "Ackley", "Rastrigin", "Griewank", "Rosenbrock"]
    dims = ["p=20", "p=100", "p=200", "p=500"]
    solvers = ["CMA-ES", "DE", "PSO", "ES-0.02", "ES-0.25", "ES-0.5", "GA-0.02", "GA-0.25", "GA-0.5"]

    left, bottom = 0.08, 0.25
    cell_w, cell_h = 0.095, 0.07

    # dimension header
    for j, d in enumerate(dims):
        ax.text(left + (j+1)*cell_w + cell_w/2, bottom + len(functions)*cell_h + 0.035,
                d, ha="center", va="center", fontsize=10, weight="bold")

    # function labels and cells
    for i, fn in enumerate(functions):
        y = bottom + (len(functions) - 1 - i)*cell_h
        ax.text(left + cell_w*0.45, y + cell_h/2, fn, ha="right", va="center", fontsize=9.5)
        for j, _ in enumerate(dims):
            x = left + (j+1)*cell_w
            color = "#EFF6FF" if i % 2 == 0 else "#F8FAFC"
            rect = FancyBboxPatch((x, y), cell_w*0.88, cell_h*0.78,
                                  boxstyle="round,pad=0.01,rounding_size=0.01",
                                  facecolor=color, edgecolor="#CBD5E1", linewidth=0.8)
            ax.add_patch(rect)
            ax.text(x + cell_w*0.44, y + cell_h*0.39, "1 problem", ha="center", va="center", fontsize=8)

    total_problems = len(functions) * len(dims)
    seeds = 30
    total_solvers = len(solvers)
    total_runs = total_problems * seeds * total_solvers

    add_box(ax, (0.62, 0.62), 0.31, 0.25, "RUN COUNT",
            f"Functions: {len(functions)}\nDimensions: {len(dims)}\nProblems: {len(functions)} × {len(dims)} = {total_problems}\nSolvers: {total_solvers}\nSeeds per pair: {seeds}\nTotal runs: {total_runs}",
            fc=COLORS["light_blue"], ec=COLORS["blue"])

    add_box(ax, (0.62, 0.31), 0.31, 0.23, "SOLVERS",
            "CMA-ES, DE, PSO\nES-0.02, ES-0.25, ES-0.5\nGA-0.02, GA-0.25, GA-0.5\n\nMỗi solver tự sinh population\nvà tối ưu cùng một problem.",
            fc=COLORS["light_green"], ec=COLORS["green"])

    ax.text(0.08, 0.15,
            "Ghi chú: nếu phiên bản của bạn chỉ bám đúng paper gốc, Scenario 1 có 6 nhóm hàm: Sphere, PA-1, PA-10, CPA, Ackley, Rastrigin.\n"
            "Nếu project của bạn thêm Griewank và Rosenbrock, tổng là 8 hàm như ma trận trên.",
            ha="left", va="top", fontsize=9.2, color=COLORS["gray"])

    savefig("02_experiment_design.png")


def fig_03_ea_loop_detail():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Inside One EA Run", ha="center", va="top", fontsize=18, weight="bold")
    ax.text(0.5, 0.915, "Một run là quá trình EA tự tạo và cải thiện population cho đến khi hết budget fitness evaluations.",
            ha="center", va="top", fontsize=10.2, color=COLORS["gray"])

    steps = [
        (0.08, 0.63, "1. Initialize", "Sinh N vector ngẫu nhiên\nx₁, x₂, ..., x_N"),
        (0.33, 0.63, "2. Evaluate", "Tính fitness\nf(x₁), f(x₂), ..., f(x_N)"),
        (0.58, 0.63, "3. Select", "Ưu tiên nghiệm\ncó fitness thấp"),
        (0.83, 0.63, "4. Generate", "Tạo offspring bằng\nmutation / crossover"),
        (0.33, 0.23, "5. Replace", "Gộp parent + offspring\ngiữ lại nghiệm tốt"),
        (0.58, 0.23, "6. Stop", "Nếu đủ 10,000 evals\ntrả best_x và best_f"),
    ]
    colors = [COLORS["light_blue"], COLORS["light_orange"], COLORS["light_green"], COLORS["light_purple"], "#FFF7ED", COLORS["light_red"]]
    edges = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"], "#EA580C", COLORS["red"]]

    for (x, y, title, body), fc, ec in zip(steps, colors, edges):
        add_box(ax, (x, y), 0.16, 0.20, title, body, fc=fc, ec=ec)

    add_arrow(ax, (0.24, 0.73), (0.33, 0.73), color=COLORS["gray"])
    add_arrow(ax, (0.49, 0.73), (0.58, 0.73), color=COLORS["gray"])
    add_arrow(ax, (0.74, 0.73), (0.83, 0.73), color=COLORS["gray"])
    add_arrow(ax, (0.83, 0.63), (0.49, 0.35), color=COLORS["purple"], text="offspring")
    add_arrow(ax, (0.49, 0.33), (0.58, 0.33), color=COLORS["gray"])
    add_arrow(ax, (0.41, 0.43), (0.33, 0.63), color=COLORS["red"], text="repeat if budget remains", text_offset=(-0.04, 0.02))

    add_box(ax, (0.08, 0.15), 0.16, 0.17, "Mutation", "x' = x + noise\nkhai thác vùng gần nghiệm tốt", fc="#F8FAFC", ec="#94A3B8")
    add_box(ax, (0.08, 0.37), 0.16, 0.17, "Crossover", "x' = mix(parent A, parent B)\nnhảy sang vùng mới", fc="#F8FAFC", ec="#94A3B8")

    ax.text(0.5, 0.055,
            "Điểm cốt lõi: EA không biết nghiệm đúng. Nó chỉ biết fitness của các nghiệm đã thử, rồi dùng chúng để tạo nghiệm mới tốt hơn.",
            ha="center", va="bottom", fontsize=10.2, color=COLORS["black"], weight="bold")

    savefig("03_ea_loop_detail.png")


def fig_04_candidate_to_fitness():
    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "From Candidate Vector to Fitness", ha="center", va="top",
            fontsize=18, weight="bold")
    ax.text(0.5, 0.91, "Trong Scenario 1, mỗi nghiệm là một vector số; hàm benchmark biến vector đó thành một fitness value.",
            ha="center", va="top", fontsize=10.2, color=COLORS["gray"])

    add_box(ax, (0.07, 0.47), 0.24, 0.30, "Candidate solution",
            "x ∈ R^p\n\nExample p = 5:\nx = [0.2, -0.7, 0.1, 0.5, -0.3]",
            fc=COLORS["light_blue"], ec=COLORS["blue"])
    add_box(ax, (0.38, 0.47), 0.24, 0.30, "Objective function",
            "Sphere: sum(xᵢ²)\nPA: ||x - target||\nCPA: min distance to targets\nRastrigin/Ackley: rugged landscape",
            fc=COLORS["light_orange"], ec=COLORS["orange"])
    add_box(ax, (0.69, 0.47), 0.24, 0.30, "Fitness value",
            "fitness = f(x)\n\nMinimization:\nlower value = better solution\n\nEA keeps improving this value",
            fc=COLORS["light_green"], ec=COLORS["green"])

    add_arrow(ax, (0.31, 0.62), (0.38, 0.62), color=COLORS["blue"], text="input")
    add_arrow(ax, (0.62, 0.62), (0.69, 0.62), color=COLORS["orange"], text="evaluate")

    # numeric example
    ax.text(0.5, 0.23,
            "Example with Sphere:  x = [2, -1, 0, 0, 0]  →  f(x) = 2² + (-1)² = 5.\n"
            "If another candidate has f(x)=0.4, EA treats it as better because this is a minimization problem.",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="#F8FAFC", ec="#CBD5E1"))

    savefig("04_candidate_to_fitness.png")


def fig_05_landscape_gallery():
    items = [
        ("Sphere\n1 global optimum", sphere, (-5.12, 5.12), np.array([[0, 0]])),
        ("PA-1\ntarget near origin", lambda X: point_aiming(X, 1.0), (-2, 4), np.array([[1, 1]])),
        ("PA-10\ntarget far from origin", lambda X: point_aiming(X, 10.0), (6, 13), np.array([[10, 10]])),
        ("CPA\n5 global targets", cpa_2d, (-3, 5), CPA_T),
        ("Ackley\nmany local optima", ackley, (-5, 5), np.array([[0, 0]])),
        ("Rastrigin\nrugged landscape", rastrigin, (-5.12, 5.12), np.array([[0, 0]])),
        ("Griewank\noscillatory", griewank, (-20, 20), np.array([[0, 0]])),
        ("Rosenbrock\nnarrow valley", rosenbrock, (-2, 2), np.array([[1, 1]])),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    axes = axes.ravel()

    for ax, (title, fn, bounds, opt) in zip(axes, items):
        X1, X2, X = make_grid(bounds, n=240)
        Z = fn(X)
        cont = ax.contourf(X1, X2, Z, levels=38, cmap="viridis")
        ax.contour(X1, X2, Z, levels=12, colors="white", linewidths=0.35, alpha=0.45)
        ax.scatter(opt[:, 0], opt[:, 1], marker="X", s=80, color="#FF2D55", edgecolors="white", linewidths=0.8)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.grid(False)

    cbar = fig.colorbar(cont, ax=axes, shrink=0.92, label="f(x)")
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Synthetic Problem Landscapes — 2D Views", fontsize=18, weight="bold")
    plt.tight_layout()
    savefig("05_landscape_gallery.png")


def fig_06_population_evolution():
    bounds = (-5.12, 5.12)
    result = run_ga_2d(rastrigin, bounds=bounds, pop_size=60, generations=60, seed=5)
    X1, X2, X = make_grid(bounds, n=300)
    Z = rastrigin(X)

    snaps = [0, 10, 30, 60]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.ravel()

    for ax, gen in zip(axes, snaps):
        pop = result["hist_pop"][gen]
        best = result["hist_best_point"][gen]
        cont = ax.contourf(X1, X2, Z, levels=42, cmap="turbo")
        ax.contour(X1, X2, Z, levels=14, colors="white", linewidths=0.35, alpha=0.55)
        ax.scatter(pop[:, 0], pop[:, 1], s=22, color="#111827", alpha=0.75, label="population")
        ax.scatter(best[0], best[1], s=160, marker="*", color="#FFD60A", edgecolors="black", linewidths=0.8, label="best candidate")
        ax.scatter(0, 0, s=120, marker="X", color="#FF2D55", edgecolors="white", linewidths=0.8, label="global optimum")
        ax.set_title(f"Generation {gen}: best fitness = {result['hist_best'][gen]:.3f}", weight="bold")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    fig.colorbar(cont, ax=axes, shrink=0.92, label="Rastrigin f(x)")
    fig.suptitle("EA Search Process: Population Evolution on Rastrigin", fontsize=18, weight="bold")
    plt.tight_layout()
    savefig("06_population_evolution.png")


def fig_07_convergence_ga_de():
    bounds = (-5.12, 5.12)
    ga = run_ga_2d(rastrigin, bounds=bounds, pop_size=60, generations=80, seed=7)
    de = run_de_2d(rastrigin, bounds=bounds, pop_size=45, generations=80, seed=7)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(ga["hist_best"], color=COLORS["purple"], linewidth=2.5, label="GA: best fitness so far")
    ax.plot(de["hist_best"], color=COLORS["green"], linewidth=2.5, label="DE: best fitness so far")
    ax.axhline(0, color=COLORS["red"], linestyle="--", linewidth=1.3, label="theoretical optimum f*=0")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness so far")
    ax.set_title("Convergence Curve: How the Output best_fitness is Produced", weight="bold")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(framealpha=0.95)

    ax.text(0.98, 0.58,
            "Lower is better\n\nThe curve decreases because\nEA keeps the best candidate\nfound so far.",
            transform=ax.transAxes, ha="right", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#CBD5E1", alpha=0.95))

    plt.tight_layout()
    savefig("07_convergence_ga_de.png")


def fig_08_output_artifacts():
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "What Comes Out After Running Experiment 1?", ha="center", va="top",
            fontsize=18, weight="bold")

    add_box(ax, (0.05, 0.53), 0.22, 0.28, "Raw run output",
            "One row per run\n\nproblem name\nsolver name\nseed\nfinal best fitness\nevaluation history",
            fc=COLORS["light_blue"], ec=COLORS["blue"])
    add_box(ax, (0.335, 0.53), 0.22, 0.28, "Aggregated results",
            "Group by:\nproblem × solver\n\nmedian fitness\nrank / NER\nEtTQ\nNoVS",
            fc=COLORS["light_green"], ec=COLORS["green"])
    add_box(ax, (0.62, 0.53), 0.22, 0.28, "Visualizations",
            "boxplots\nconvergence curves\nlandscape slices\nheatmaps\nsummary tables",
            fc=COLORS["light_orange"], ec=COLORS["orange"])
    add_box(ax, (0.43, 0.14), 0.30, 0.23, "Interpretation for report",
            "Which EA finds lower fitness?\nWhich EA converges faster?\nWhich landscape is difficult?\nDoes p make the problem harder?",
            fc=COLORS["light_red"], ec=COLORS["red"])

    add_arrow(ax, (0.27, 0.67), (0.335, 0.67), color=COLORS["blue"])
    add_arrow(ax, (0.555, 0.67), (0.62, 0.67), color=COLORS["green"])
    add_arrow(ax, (0.73, 0.53), (0.58, 0.37), color=COLORS["orange"])

    ax.text(0.05, 0.06,
            "Trên máy của bạn, nên lưu kết quả vào thư mục riêng, ví dụ: results/scenario1/ và figures/scenario1/.\n"
            "Synthetic problems không cần GPU; CPU i7-12700H và RAM 32 GB là đủ để chạy và vẽ hình minh họa.",
            ha="left", va="bottom", fontsize=9.4, color=COLORS["gray"])

    savefig("08_output_artifacts.png")


def main():
    print("Creating Experiment 1 workflow visualizations...")
    fig_01_workflow_overview()
    fig_02_experiment_design()
    fig_03_ea_loop_detail()
    fig_04_candidate_to_fitness()
    fig_05_landscape_gallery()
    fig_06_population_evolution()
    fig_07_convergence_ga_de()
    fig_08_output_artifacts()
    print("\nDone.")
    print("Output folder:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
