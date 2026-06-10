"""
visualize_benchmarks.py
========================
Trực quan hóa 7 hàm benchmark tối ưu hóa:
  1. Sphere
  2. Point Aiming (PA-1, PA-10)
  3. Circular Point Aiming (CPA)
  4. Ackley
  5. Rastrigin
  6. Griewank
  7. Rosenbrock

Xuất ra:
  - Đồ thị 3D (surface) cho từng hàm (n=2 chiều)
  - Lát cắt 2D theo chiều p = 20, 100, 200, 500
  - Tóm tắt điểm tối ưu toàn cục & local optima (nếu có)
  - Tất cả ảnh lưu vào thư mục output_imgs/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  Thư mục đầu ra
# ─────────────────────────────────────────────
OUTPUT_DIR = "output_imgs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Matplotlib style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})

DIMS = [20, 100, 200, 500]   # chiều thí nghiệm


# ═══════════════════════════════════════════════════════════════
# 1.  Định nghĩa hàm benchmark
# ═══════════════════════════════════════════════════════════════

def sphere(x: np.ndarray) -> float:
    """f(x) = sum(xi^2).  Global min = 0 at x*=(0,...,0)."""
    return float(np.sum(x ** 2))


def point_aiming(x: np.ndarray, sigma: float = 1.0) -> float:
    """
    Point Aiming (PA).
    f(x) = -exp( -||x - target||^2 / (2*sigma^2) )
    target = (1, 1, ..., 1).  Global min → -1 at x=target.
    sigma=1 → PA-1,  sigma=10 → PA-10.
    """
    target = np.ones_like(x)
    return float(-np.exp(-np.sum((x - target) ** 2) / (2 * sigma ** 2)))


def circular_point_aiming(x: np.ndarray, r: float = 1.0) -> float:
    """
    Circular Point Aiming (CPA).
    f(x) = min over theta of ||x - target(theta)||^2
    Circular attractor lives in the first two dimensions:
        target = (r*cos(theta), r*sin(theta), 0, ..., 0).
    Global min = 0 on the circle; no unique global minimiser.
    """
    r_x = np.sqrt(x[0] ** 2 + x[1] ** 2)   # radius of (x1, x2)
    rest = np.sum(x[2:] ** 2) if len(x) > 2 else 0.0
    radial_dist = (r_x - r) ** 2
    return float(radial_dist + rest)


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> float:
    """
    Ackley function.
    Global min = 0 at x*=(0,...,0).
    Many local optima (highly multimodal).
    """
    n = len(x)
    sum_sq   = np.sum(x ** 2)
    sum_cos  = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return float(term1 + term2 + a + np.e)


def rastrigin(x: np.ndarray, A: float = 10.0) -> float:
    """
    Rastrigin function.
    Global min = 0 at x*=(0,...,0).
    Large number of local optima (~spaced 1 apart).
    """
    n = len(x)
    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def griewank(x: np.ndarray) -> float:
    """
    Griewank function.
    Global min = 0 at x*=(0,...,0).
    Local optima present but diminish as dimension grows.
    """
    sum_sq = np.sum(x ** 2) / 4000.0
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(sum_sq - prod_cos + 1.0)


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock (banana) function.
    Global min = 0 at x*=(1,...,1).
    Unimodal in most cases (slight local optima for n>=4).
    """
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


# ─────────────────────────────────────────────
# Metadata cho từng hàm
# ─────────────────────────────────────────────
FUNCTIONS = [
    {
        "name": "Sphere",
        "fn":   sphere,
        "domain": (-5.12, 5.12),
        "x_star": "0 vector",
        "f_star": 0.0,
        "local_optima": "None (unimodal)",
        "type": "Unimodal",
        "color": cm.plasma,
    },
    {
        "name": "Point Aiming (σ=1)",
        "fn":   lambda x: point_aiming(x, sigma=1.0),
        "domain": (-3.0, 5.0),
        "x_star": "(1, 1, ..., 1)",
        "f_star": -1.0,
        "local_optima": "None (unimodal, flat far from target)",
        "type": "Unimodal",
        "color": cm.viridis,
    },
    {
        "name": "Point Aiming (σ=10)",
        "fn":   lambda x: point_aiming(x, sigma=10.0),
        "domain": (-3.0, 5.0),
        "x_star": "(1, 1, ..., 1)",
        "f_star": -1.0,
        "local_optima": "None (unimodal, wider basin)",
        "type": "Unimodal",
        "color": cm.magma,
    },
    {
        "name": "Circular Point Aiming",
        "fn":   circular_point_aiming,
        "domain": (-2.5, 2.5),
        "x_star": "Circle x1²+x2²=1, x3=…=xn=0",
        "f_star": 0.0,
        "local_optima": "Continuous manifold of minima (circle)",
        "type": "Multi-modal",
        "color": cm.cool,
    },
    {
        "name": "Ackley",
        "fn":   ackley,
        "domain": (-32.768, 32.768),
        "x_star": "(0, 0, ..., 0)",
        "f_star": 0.0,
        "local_optima": "Many (exponential in n) — highly multi-modal",
        "type": "Multi-modal + Local optima",
        "color": cm.inferno,
    },
    {
        "name": "Rastrigin",
        "fn":   rastrigin,
        "domain": (-5.12, 5.12),
        "x_star": "(0, 0, ..., 0)",
        "f_star": 0.0,
        "local_optima": f"~(2×5.12/1)^n local minima spaced ~1 apart",
        "type": "Multi-modal + Local optima",
        "color": cm.RdYlGn_r,
    },
    {
        "name": "Griewank",
        "fn":   griewank,
        "domain": (-600.0, 600.0),
        "x_star": "(0, 0, ..., 0)",
        "f_star": 0.0,
        "local_optima": "Many in low dim; diminish at high dim",
        "type": "Multi-modal + Local optima",
        "color": cm.YlOrRd,
    },
    {
        "name": "Rosenbrock",
        "fn":   rosenbrock,
        "domain": (-2.048, 2.048),
        "x_star": "(1, 1, ..., 1)",
        "f_star": 0.0,
        "local_optima": "Virtually none (unimodal, narrow curved valley)",
        "type": "Unimodal",
        "color": cm.cividis,
    },
]


# ═══════════════════════════════════════════════════════════════
# 2.  Đồ thị 3D — mỗi hàm một ảnh
# ═══════════════════════════════════════════════════════════════

def plot_3d(info: dict):
    """Vẽ surface plot 3D với n=2."""
    fn   = info["fn"]
    lo, hi = info["domain"]
    name   = info["name"]
    cmap   = info["color"]

    N = 300
    xs = np.linspace(lo, hi, N)
    ys = np.linspace(lo, hi, N)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[fn(np.array([xi, yi])) for xi in xs] for yi in ys])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ls = LightSource(azdeg=225, altdeg=45)
    rgb = ls.shade(Z, cmap=cmap, vert_exag=0.1, blend_mode="soft")
    surf = ax.plot_surface(X, Y, Z, facecolors=rgb, linewidth=0,
                           antialiased=True, shade=False, alpha=0.92)
    ax.contourf(X, Y, Z, zdir="z", offset=Z.min() - (Z.max()-Z.min())*0.05,
                cmap=cmap, alpha=0.4, levels=30)

    ax.set_title(f"{name}\n(n=2, domain=[{lo}, {hi}])", fontsize=14, pad=16, weight="bold")
    ax.set_xlabel("x₁", labelpad=8)
    ax.set_ylabel("x₂", labelpad=8)
    ax.set_zlabel("f(x)", labelpad=8)
    ax.tick_params(labelsize=7)

    # Điểm tối ưu
    x_star_label = info["x_star"]
    f_star       = info["f_star"]
    ax.text2D(0.02, 0.97,
              f"Global min ≈ {f_star:.4g}\nat x* = {x_star_label}\nType: {info['type']}\nLocal optima: {info['local_optima']}",
              transform=ax.transAxes, fontsize=8, va="top",
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(Z)
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=12, label="f(x)")

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"3D_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [3D]  Saved → {fname}")


# ═══════════════════════════════════════════════════════════════
# 3.  Lát cắt 2D theo chiều p = 20, 100, 200, 500
# ═══════════════════════════════════════════════════════════════

def _cross_section_1d(fn, domain, dim, n_pts=500):
    """
    Lát cắt 1D: giữ tất cả biến = 0 trừ x₁ (quét qua domain).
    Trả về (t, values).
    """
    lo, hi = domain
    t = np.linspace(lo, hi, n_pts)
    vals = []
    for ti in t:
        x = np.zeros(dim)
        x[0] = ti
        vals.append(fn(x))
    return t, np.array(vals)


def count_local_minima_1d(vals: np.ndarray) -> int:
    """Đếm số cực tiểu (local minima) trong chuỗi 1D."""
    count = 0
    for i in range(1, len(vals) - 1):
        if vals[i] < vals[i-1] and vals[i] < vals[i+1]:
            count += 1
    return count


def plot_cross_sections(info: dict):
    """
    Vẽ 4 lát cắt 2D (p=20,100,200,500) trong một figure.
    """
    fn   = info["fn"]
    name = info["name"]

    palette = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    fig.suptitle(f"2D Cross-Section — {name}\n(all variables = 0 except x₁)",
                 fontsize=14, weight="bold", y=1.01)

    for idx, (p, color) in enumerate(zip(DIMS, palette)):
        ax = axes[idx]
        t, vals = _cross_section_1d(fn, info["domain"], dim=p)
        n_local = count_local_minima_1d(vals)
        global_min_idx = int(np.argmin(vals))
        global_min_val = vals[global_min_idx]
        global_min_x   = t[global_min_idx]

        ax.plot(t, vals, color=color, linewidth=1.8, label=f"p={p}")
        ax.axhline(global_min_val, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.scatter([global_min_x], [global_min_val], color="red", zorder=5, s=60,
                   label=f"Min ≈ {global_min_val:.4g} @ x={global_min_x:.3g}")

        ax.set_title(f"p = {p}  |  Local minima detected: {n_local}", fontsize=11, weight="bold")
        ax.set_xlabel("x₁")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(fontsize=8, loc="upper right")

        # annotation info
        info_txt = (f"Global min ≈ {global_min_val:.5g}\n"
                    f"x₁ ≈ {global_min_x:.4g}\n"
                    f"1D local minima: {n_local}")
        ax.text(0.02, 0.98, info_txt, transform=ax.transAxes,
                fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow", ec="gray", alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname = os.path.join(OUTPUT_DIR, f"2D_cross_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [2D]  Saved → {fname}")


# ═══════════════════════════════════════════════════════════════
# 4.  Bảng tóm tắt tối ưu
# ═══════════════════════════════════════════════════════════════

def compute_summary_table(functions: list, dims: list):
    """
    Tính giá trị hàm tại điểm tối ưu lý thuyết và
    giá trị nhỏ nhất thực sự tìm được bằng quét 1D lát cắt.
    """
    rows = []
    for info in functions:
        fn   = info["fn"]
        name = info["name"]
        for p in dims:
            t, vals = _cross_section_1d(fn, info["domain"], dim=p, n_pts=1000)
            n_local  = count_local_minima_1d(vals)
            obs_min  = float(np.min(vals))
            rows.append({
                "Function": name,
                "p": p,
                "Type": info["type"],
                "Theoretical x*": info["x_star"],
                "Theoretical f*": info["f_star"],
                "Observed 1D min": round(obs_min, 6),
                "1D Local Minima": n_local,
                "Note": info["local_optima"],
            })
    return rows


def plot_summary_table(rows: list):
    """Vẽ bảng tóm tắt và lưu ảnh."""
    import pandas as pd
    df = pd.DataFrame(rows)

    n_rows, n_cols = df.shape
    fig_h = max(4, 0.38 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")

    col_widths = [0.13, 0.04, 0.13, 0.14, 0.08, 0.10, 0.10, 0.28]
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Header style
    header_color = "#2B4590"
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", weight="bold")

    # Alternate row colors
    colors = ["#EFF3FF", "#FFFFFF"]
    type_colors = {
        "Unimodal":                      "#D4EDDA",
        "Multi-modal":                   "#FFF3CD",
        "Multi-modal + Local optima":    "#F8D7DA",
    }
    for i in range(1, n_rows + 1):
        row_type = df.iloc[i-1]["Type"]
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_facecolor(type_colors.get(row_type, colors[(i-1) % 2]))

    ax.set_title("Benchmark Functions — Optimum Summary",
                 fontsize=14, weight="bold", pad=14)

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=v, edgecolor="gray", label=k)
        for k, v in type_colors.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              bbox_to_anchor=(1.0, -0.02), fontsize=9,
              title="Function type", title_fontsize=9)

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "Summary_Optima_Table.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [TABLE] Saved → {fname}")


# ═══════════════════════════════════════════════════════════════
# 5.  Heatmap giá trị hàm theo chiều p (tổng quan)
# ═══════════════════════════════════════════════════════════════

def plot_min_vs_dim(rows: list):
    """Bar chart: observed 1D min theo (function, p)."""
    import pandas as pd
    df = pd.DataFrame(rows)

    func_names = df["Function"].unique().tolist()
    x = np.arange(len(func_names))
    width = 0.18
    offsets = np.linspace(-0.27, 0.27, len(DIMS))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (p, off, col) in enumerate(zip(DIMS, offsets, palette)):
        sub = df[df["p"] == p].set_index("Function")
        vals = [sub.loc[f, "Observed 1D min"] if f in sub.index else 0 for f in func_names]
        ax.bar(x + off, vals, width, label=f"p={p}", color=col, alpha=0.82, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(func_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Observed 1D minimum value")
    ax.set_title("Observed 1D Minimum per Function × Dimension p",
                 fontsize=13, weight="bold")
    ax.legend(title="Dimension p", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "Min_vs_Dimension.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [BAR]  Saved → {fname}")


def plot_local_minima_heatmap(rows: list):
    """Heatmap số local minima 1D theo (function, p)."""
    import pandas as pd
    df = pd.DataFrame(rows)

    pivot = df.pivot(index="Function", columns="p", values="1D Local Minima")

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([f"p={p}" for p in DIMS], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Number of 1D Local Minima per Function × Dimension",
                 fontsize=13, weight="bold")
    plt.colorbar(im, ax=ax, label="# local minima (1D cross-section)")

    for i in range(len(pivot.index)):
        for j in range(len(DIMS)):
            val = pivot.values[i, j]
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=10, color="black" if val < pivot.values.max()*0.6 else "white",
                    weight="bold")

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "Local_Minima_Heatmap.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [HMAP] Saved → {fname}")


# ═══════════════════════════════════════════════════════════════
# 6.  Overlay: tất cả hàm cùng chiều p (normalized)
# ═══════════════════════════════════════════════════════════════

def plot_all_functions_overlay():
    """Một grid plot: mỗi hàng là một chiều p, mỗi cột là một hàm."""
    n_fn = len(FUNCTIONS)
    n_p  = len(DIMS)
    fig, axes = plt.subplots(n_p, n_fn, figsize=(3.0 * n_fn, 3.2 * n_p))

    palette_fn = plt.get_cmap("tab10")

    for row, p in enumerate(DIMS):
        for col, info in enumerate(FUNCTIONS):
            ax = axes[row, col]
            fn = info["fn"]
            t, vals = _cross_section_1d(fn, info["domain"], dim=p, n_pts=400)
            color = palette_fn(col / n_fn)

            ax.plot(t, vals, color=color, linewidth=1.4)
            # mark global min
            gmin_val = np.min(vals)
            gmin_x   = t[np.argmin(vals)]
            ax.scatter([gmin_x], [gmin_val], color="red", s=25, zorder=5)
            ax.set_title(info["name"] if row == 0 else "", fontsize=8, weight="bold")
            ax.set_ylabel(f"p={p}" if col == 0 else "", fontsize=9)
            ax.grid(True, alpha=0.2, linestyle=":")
            ax.tick_params(labelsize=6)

    fig.suptitle("All Benchmark Functions — 2D Cross-Sections by Dimension p",
                 fontsize=14, weight="bold", y=1.005)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "All_Functions_Cross_Section_Grid.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [GRID] Saved → {fname}")


# ═══════════════════════════════════════════════════════════════
# 7.  Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Benchmark Function Visualizer")
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

    # --- 3D surface plots ---
    print("\n[1/5] Rendering 3D surface plots …")
    for info in FUNCTIONS:
        print(f"  → {info['name']}")
        plot_3d(info)

    # --- 2D cross-sections per function ---
    print("\n[2/5] Rendering 2D cross-sections per function …")
    for info in FUNCTIONS:
        print(f"  → {info['name']}")
        plot_cross_sections(info)

    # --- compute summary ---
    print("\n[3/5] Computing optimum summary table …")
    rows = compute_summary_table(FUNCTIONS, DIMS)

    # Print table to console
    header = f"{'Function':<30} {'p':>5} {'f*':>10} {'1D min':>12} {'Local min':>10}  Type"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"  {r['Function']:<28} {r['p']:>5} {r['Theoretical f*']:>10.4g}"
              f" {r['Observed 1D min']:>12.5g} {r['1D Local Minima']:>10}  {r['Type']}")

    plot_summary_table(rows)

    # --- bar chart + heatmap ---
    print("\n[4/5] Rendering bar chart & heatmap …")
    plot_min_vs_dim(rows)
    plot_local_minima_heatmap(rows)

    # --- grid overlay ---
    print("\n[5/5] Rendering full cross-section grid …")
    plot_all_functions_overlay()

    print("\n" + "=" * 60)
    print(f"  ✓ All plots saved in:  {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
