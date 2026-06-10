"""
visualize_benchmarks_fixed.py
============================= 
Trực quan hóa các hàm benchmark tối ưu hóa:
  1. Sphere
  2. Point Aiming (PA-1, PA-10)
  3. Circular Point Aiming (CPA)
  4. Ackley
  5. Rastrigin
  6. Griewank
  7. Rosenbrock

Các điểm đã sửa so với bản cũ:
  - Phân biệt rõ 3D surface, 2D slice và 1D cross-section.
  - 2D/1D slice của PA và Rosenbrock đi qua nghiệm tối ưu (1,...,1).
  - CPA được ghi là multiple global optima trên một đường tròn, không phải local optima rời rạc.
  - Rosenbrock được ghi là unimodal, khó vì narrow curved valley.
  - Griewank được ghi là multimodal do cosine product term.

Xuất ra thư mục output_imgs/:
  - 3D surface plot cho từng hàm với n=2.
  - 2D contour slice cho từng hàm ở p = 20, 100, 200, 500.
  - 1D cross-section cho từng hàm ở p = 20, 100, 200, 500.
  - Bảng tóm tắt optimum.
  - Heatmap số cực tiểu cục bộ phát hiện trên 1D cross-section.

Lưu ý:
  PA và CPA không phải benchmark phổ quát như Sphere/Ackley/Rastrigin.
  Ở đây PA/CPA được cài theo công thức minh họa nhất quán với ý nghĩa:
    PA  = nhắm tới một điểm target.
    CPA = nhắm tới một vòng tròn target.
  Nếu tài liệu/paper gốc của bạn định nghĩa PA/CPA khác, hãy thay hai hàm
  point_aiming() và circular_point_aiming() bằng công thức chính thức.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────
OUTPUT_DIR = Path("output_imgs")
OUTPUT_DIR.mkdir(exist_ok=True)

DIMS = [20, 100, 200, 500]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})


def safe_name(name: str) -> str:
    """Create a filesystem-safe name."""
    name = name.replace("σ", "sigma")
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return name.strip("_")


# ═══════════════════════════════════════════════════════════════
# 1. Benchmark functions
# All functions accept x with shape (..., p) and return shape (...,)
# ═══════════════════════════════════════════════════════════════

def sphere(x: np.ndarray) -> np.ndarray:
    """Sphere: f(x)=sum_i x_i^2. Global min 0 at x=(0,...,0)."""
    return np.sum(x**2, axis=-1)


def point_aiming(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Point Aiming (PA).

    f(x) = -exp(-||x - target||^2 / (2*sigma^2)), target=(1,...,1).

    Global minimum: -1 at x=(1,...,1).
    sigma=1  -> PA-1:  narrower basin.
    sigma=10 -> PA-10: wider basin.
    """
    target = np.ones(x.shape[-1])
    dist2 = np.sum((x - target) ** 2, axis=-1)
    return -np.exp(-dist2 / (2 * sigma**2))


def circular_point_aiming(x: np.ndarray, r: float = 1.0) -> np.ndarray:
    """
    Circular Point Aiming (CPA).

    The target is a circle in the first two dimensions:
        target(theta) = (r*cos(theta), r*sin(theta), 0, ..., 0)

    Equivalent closed-form distance to that circle:
        f(x) = (sqrt(x1^2+x2^2)-r)^2 + sum_{i=3}^p x_i^2

    Global minimum: 0 on the circle x1^2+x2^2=r^2 and x3=...=xp=0.
    Therefore, CPA has multiple/global-optimal solutions, not a unique optimum.
    """
    radius_xy = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
    rest = np.sum(x[..., 2:] ** 2, axis=-1) if x.shape[-1] > 2 else 0.0
    return (radius_xy - r) ** 2 + rest


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> np.ndarray:
    """Ackley. Global min 0 at x=(0,...,0). Multimodal with many local optima."""
    p = x.shape[-1]
    sum_sq = np.sum(x**2, axis=-1)
    sum_cos = np.sum(np.cos(c * x), axis=-1)
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / p))
    term2 = -np.exp(sum_cos / p)
    return term1 + term2 + a + np.e


def rastrigin(x: np.ndarray, A: float = 10.0) -> np.ndarray:
    """Rastrigin. Global min 0 at x=(0,...,0). Multimodal with many local optima."""
    p = x.shape[-1]
    return A * p + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=-1)


def griewank(x: np.ndarray) -> np.ndarray:
    """Griewank. Global min 0 at x=(0,...,0). Multimodal due to cosine product term."""
    p = x.shape[-1]
    indices = np.arange(1, p + 1)
    sum_term = np.sum(x**2, axis=-1) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(indices)), axis=-1)
    return 1.0 + sum_term - prod_term


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """
    Rosenbrock banana function.

    Global minimum: 0 at x=(1,...,1).
    Treated as unimodal in the standard benchmark setting; difficult because
    the optimum lies in a narrow, curved valley.
    """
    return np.sum(100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2, axis=-1)


@dataclass(frozen=True)
class Benchmark:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    domain: tuple[float, float]
    optimum_text: str
    f_star: float
    landscape_type: str
    local_optima_text: str
    slice_base: str  # "zero" or "one"
    cmap: object


FUNCTIONS = [
    Benchmark(
        name="Sphere",
        fn=sphere,
        domain=(-5.12, 5.12),
        optimum_text="x* = (0, ..., 0)",
        f_star=0.0,
        landscape_type="Unimodal",
        local_optima_text="None",
        slice_base="zero",
        cmap=cm.plasma,
    ),
    Benchmark(
        name="Point Aiming (sigma=1)",
        fn=lambda x: point_aiming(x, sigma=1.0),
        domain=(-3.0, 5.0),
        optimum_text="x* = (1, ..., 1)",
        f_star=-1.0,
        landscape_type="Unimodal",
        local_optima_text="None; flat far from target",
        slice_base="one",
        cmap=cm.viridis,
    ),
    Benchmark(
        name="Point Aiming (sigma=10)",
        fn=lambda x: point_aiming(x, sigma=10.0),
        domain=(-3.0, 5.0),
        optimum_text="x* = (1, ..., 1)",
        f_star=-1.0,
        landscape_type="Unimodal",
        local_optima_text="None; wider basin than sigma=1",
        slice_base="one",
        cmap=cm.magma,
    ),
    Benchmark(
        name="Circular Point Aiming",
        fn=circular_point_aiming,
        domain=(-2.5, 2.5),
        optimum_text="x1^2+x2^2=1, x3=...=xp=0",
        f_star=0.0,
        landscape_type="Multiple global optima",
        local_optima_text="No isolated local optima; global minima form a circle",
        slice_base="zero",
        cmap=cm.cool,
    ),
    Benchmark(
        name="Ackley",
        fn=ackley,
        domain=(-32.768, 32.768),
        optimum_text="x* = (0, ..., 0)",
        f_star=0.0,
        landscape_type="Multimodal + local optima",
        local_optima_text="Many local optima",
        slice_base="zero",
        cmap=cm.inferno,
    ),
    Benchmark(
        name="Rastrigin",
        fn=rastrigin,
        domain=(-5.12, 5.12),
        optimum_text="x* = (0, ..., 0)",
        f_star=0.0,
        landscape_type="Multimodal + local optima",
        local_optima_text="Many regularly distributed local optima",
        slice_base="zero",
        cmap=cm.RdYlGn_r,
    ),
    Benchmark(
        name="Griewank",
        fn=griewank,
        domain=(-600.0, 600.0),
        optimum_text="x* = (0, ..., 0)",
        f_star=0.0,
        landscape_type="Multimodal + local optima",
        local_optima_text="Many local optima caused by cosine product term",
        slice_base="zero",
        cmap=cm.YlOrRd,
    ),
    Benchmark(
        name="Rosenbrock",
        fn=rosenbrock,
        domain=(-2.048, 2.048),
        optimum_text="x* = (1, ..., 1)",
        f_star=0.0,
        landscape_type="Unimodal",
        local_optima_text="No typical isolated local optima; narrow curved valley",
        slice_base="one",
        cmap=cm.cividis,
    ),
]


# ═══════════════════════════════════════════════════════════════
# 2. Slice helpers
# ═══════════════════════════════════════════════════════════════

def base_point(info: Benchmark, p: int) -> np.ndarray:
    """
    Base point for slices.

    zero-based functions: other variables fixed at 0.
    one-based functions: other variables fixed at 1 so the slice passes the global optimum.
    """
    if info.slice_base == "one":
        return np.ones(p)
    return np.zeros(p)


def make_2d_slice(info: Benchmark, p: int, n_pts: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    True 2D slice:
      vary x1 and x2;
      fix x3,...,xp at a base value chosen so the slice is meaningful.

    For PA and Rosenbrock, x3,...,xp are fixed at 1.
    For Sphere/Ackley/Rastrigin/Griewank/CPA, x3,...,xp are fixed at 0.
    """
    lo, hi = info.domain
    x1 = np.linspace(lo, hi, n_pts)
    x2 = np.linspace(lo, hi, n_pts)
    X1, X2 = np.meshgrid(x1, x2)

    X = np.zeros((n_pts, n_pts, p), dtype=float)
    X[:] = base_point(info, p)
    X[..., 0] = X1
    X[..., 1] = X2
    Z = info.fn(X)
    return X1, X2, Z


def make_1d_cross_section(info: Benchmark, p: int, n_pts: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    1D cross-section:
      vary x1;
      fix x2,...,xp at a base value chosen so the slice passes the optimum.

    This is intentionally called 1D cross-section, not 2D slice.
    """
    lo, hi = info.domain
    t = np.linspace(lo, hi, n_pts)
    X = np.zeros((n_pts, p), dtype=float)
    X[:] = base_point(info, p)
    X[:, 0] = t
    vals = info.fn(X)
    return t, vals


def count_local_minima_1d(vals: np.ndarray) -> int:
    """Count strict local minima along a 1D sampled curve."""
    vals = np.asarray(vals)
    return int(np.sum((vals[1:-1] < vals[:-2]) & (vals[1:-1] < vals[2:])))


# ═══════════════════════════════════════════════════════════════
# 3. Plots
# ═══════════════════════════════════════════════════════════════

def plot_3d_surface(info: Benchmark, n_pts: int = 260) -> None:
    """Plot a 3D surface for p=2."""
    X1, X2, Z = make_2d_slice(info, p=2, n_pts=n_pts)
    lo, hi = info.domain

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ls = LightSource(azdeg=225, altdeg=45)
    rgb = ls.shade(Z, cmap=info.cmap, vert_exag=0.1, blend_mode="soft")

    z_offset = float(np.nanmin(Z) - (np.nanmax(Z) - np.nanmin(Z)) * 0.05)
    ax.plot_surface(X1, X2, Z, facecolors=rgb, linewidth=0, antialiased=True, shade=False, alpha=0.94)
    ax.contourf(X1, X2, Z, zdir="z", offset=z_offset, cmap=info.cmap, alpha=0.42, levels=35)

    ax.set_title(f"{info.name}\n3D surface, p=2, domain=[{lo}, {hi}]", fontsize=14, pad=16, weight="bold")
    ax.set_xlabel("x1", labelpad=8)
    ax.set_ylabel("x2", labelpad=8)
    ax.set_zlabel("f(x)", labelpad=8)
    ax.tick_params(labelsize=7)

    ax.text2D(
        0.02,
        0.97,
        f"Global min: {info.f_star:.4g}\n{info.optimum_text}\nType: {info.landscape_type}\nLocal optima: {info.local_optima_text}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.86),
    )

    m = cm.ScalarMappable(cmap=info.cmap)
    m.set_array(Z)
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=12, label="f(x)")

    plt.tight_layout()
    fname = OUTPUT_DIR / f"3D_{safe_name(info.name)}.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [3D] Saved -> {fname}")


def plot_2d_slices(info: Benchmark, n_pts: int = 220) -> None:
    """Plot true 2D contour slices for p=20,100,200,500."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()
    fig.suptitle(
        f"2D contour slices — {info.name}\nVary x1 and x2; fix remaining variables at the slice base",
        fontsize=14,
        weight="bold",
        y=1.02,
    )

    for ax, p in zip(axes, DIMS):
        X1, X2, Z = make_2d_slice(info, p=p, n_pts=n_pts)
        contour = ax.contourf(X1, X2, Z, levels=70, cmap=info.cmap)
        ax.set_title(f"p = {p}", fontsize=11, weight="bold")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, alpha=0.18, linestyle=":")
        fig.colorbar(contour, ax=ax, shrink=0.85)

        # Mark representative global optimum if visible in this 2D slice.
        if info.name.startswith("Circular Point Aiming"):
            circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=1.5, linestyle="--")
            ax.add_patch(circle)
            ax.text(0.02, 0.98, "Global minima: circle", transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
        elif info.slice_base == "one":
            ax.scatter([1], [1], s=55, marker="x", c="black", label="global optimum")
            ax.legend(fontsize=8, loc="upper right")
        else:
            ax.scatter([0], [0], s=55, marker="x", c="black", label="global optimum")
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fname = OUTPUT_DIR / f"2D_slice_{safe_name(info.name)}.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [2D] Saved -> {fname}")


def plot_1d_cross_sections(info: Benchmark) -> None:
    """Plot 1D cross-sections for p=20,100,200,500."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    fig.suptitle(
        f"1D cross-sections — {info.name}\nVary x1; fix remaining variables at the slice base",
        fontsize=14,
        weight="bold",
        y=1.01,
    )

    for ax, p in zip(axes, DIMS):
        t, vals = make_1d_cross_section(info, p=p, n_pts=1200)
        n_local = count_local_minima_1d(vals)
        min_idx = int(np.argmin(vals))
        min_x = float(t[min_idx])
        min_val = float(vals[min_idx])

        ax.plot(t, vals, linewidth=1.7, label=f"p={p}")
        ax.scatter([min_x], [min_val], s=55, zorder=5, label=f"sampled min ≈ {min_val:.4g}")
        ax.axhline(min_val, linewidth=0.8, linestyle="--", alpha=0.65)
        ax.set_title(f"p = {p} | sampled 1D local minima: {n_local}", fontsize=11, weight="bold")
        ax.set_xlabel("x1")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(fontsize=8, loc="best")

        ax.text(
            0.02,
            0.98,
            f"Sampled min ≈ {min_val:.5g}\nx1 ≈ {min_x:.4g}\n1D local minima: {n_local}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow", ec="gray", alpha=0.9),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname = OUTPUT_DIR / f"1D_cross_{safe_name(info.name)}.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [1D] Saved -> {fname}")


# ═══════════════════════════════════════════════════════════════
# 4. Summary table and aggregate plots
# ═══════════════════════════════════════════════════════════════

def compute_summary_rows() -> list[dict]:
    rows = []
    for info in FUNCTIONS:
        for p in DIMS:
            t, vals = make_1d_cross_section(info, p=p, n_pts=1600)
            rows.append({
                "Function": info.name,
                "p": p,
                "Type": info.landscape_type,
                "Theoretical optimum": info.optimum_text,
                "Theoretical f*": info.f_star,
                "Sampled 1D min": round(float(np.min(vals)), 6),
                "Sampled 1D local minima": count_local_minima_1d(vals),
                "Note": info.local_optima_text,
            })
    return rows


def plot_summary_table(rows: list[dict]) -> None:
    import pandas as pd
    from matplotlib.patches import Patch

    df = pd.DataFrame(rows)
    n_rows, n_cols = df.shape
    fig_h = max(5, 0.42 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(19, fig_h))
    ax.axis("off")

    col_widths = [0.14, 0.04, 0.15, 0.18, 0.08, 0.10, 0.12, 0.19]
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)

    header_color = "#2B4590"
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", weight="bold")

    type_colors = {
        "Unimodal": "#D4EDDA",
        "Multiple global optima": "#D1ECF1",
        "Multimodal + local optima": "#F8D7DA",
    }
    for i in range(1, n_rows + 1):
        row_type = df.iloc[i - 1]["Type"]
        for j in range(n_cols):
            table[i, j].set_facecolor(type_colors.get(row_type, "#FFFFFF"))

    ax.set_title("Benchmark functions — optimum and sampled 1D cross-section summary", fontsize=14, weight="bold", pad=14)

    legend_patches = [Patch(facecolor=v, edgecolor="gray", label=k) for k, v in type_colors.items()]
    ax.legend(handles=legend_patches, loc="lower right", bbox_to_anchor=(1.0, -0.02), fontsize=9, title="Landscape type")

    plt.tight_layout()
    fname = OUTPUT_DIR / "Summary_Optima_Table.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [TABLE] Saved -> {fname}")


def plot_local_minima_heatmap(rows: list[dict]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Function", columns="p", values="Sampled 1D local minima")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([f"p={p}" for p in DIMS], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Sampled 1D local minima per function and dimension", fontsize=13, weight="bold")
    plt.colorbar(im, ax=ax, label="# local minima along sampled 1D cross-section")

    max_val = max(1, int(np.max(pivot.values)))
    for i in range(len(pivot.index)):
        for j in range(len(DIMS)):
            val = int(pivot.values[i, j])
            ax.text(j, i, str(val), ha="center", va="center", fontsize=10,
                    color="black" if val < max_val * 0.6 else "white", weight="bold")

    plt.tight_layout()
    fname = OUTPUT_DIR / "Sampled_1D_Local_Minima_Heatmap.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [HMAP] Saved -> {fname}")


def plot_all_functions_1d_grid() -> None:
    """Grid of 1D cross-sections: rows are dimensions, columns are functions."""
    n_fn = len(FUNCTIONS)
    n_p = len(DIMS)
    fig, axes = plt.subplots(n_p, n_fn, figsize=(3.0 * n_fn, 3.1 * n_p))

    for row, p in enumerate(DIMS):
        for col, info in enumerate(FUNCTIONS):
            ax = axes[row, col]
            t, vals = make_1d_cross_section(info, p=p, n_pts=700)
            min_idx = int(np.argmin(vals))
            ax.plot(t, vals, linewidth=1.3)
            ax.scatter([t[min_idx]], [vals[min_idx]], s=22, zorder=5)
            if row == 0:
                ax.set_title(info.name, fontsize=8, weight="bold")
            if col == 0:
                ax.set_ylabel(f"p={p}", fontsize=9)
            ax.grid(True, alpha=0.2, linestyle=":")
            ax.tick_params(labelsize=6)

    fig.suptitle("All benchmark functions — 1D cross-sections by dimension p", fontsize=14, weight="bold", y=1.005)
    plt.tight_layout()
    fname = OUTPUT_DIR / "All_Functions_1D_Cross_Section_Grid.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [GRID] Saved -> {fname}")





def add_landscape_markers(ax, info: Benchmark, lo: float, hi: float) -> None:
    """
    Add global/local-optimum markers on a 2D slice.

    Blue marker  : global optimum / global-optimum set.
    Red markers  : illustrative local basin centers for highly multimodal functions.

    For Rastrigin/Ackley the red markers are visual aids on the 2D slice; they are not
    produced by a numerical optimizer. For Griewank local minima are not placed because
    their locations are less regular on this slice.
    """
    name = info.name

    # Global optima / global optimum set
    if name.startswith("Circular Point Aiming"):
        circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=2.0, linestyle="--", color="white")
        ax.add_patch(circle)
        ax.text(0.02, 0.98, "global optima: circle", transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))
    elif info.slice_base == "one":
        if lo <= 1 <= hi:
            ax.scatter([1], [1], s=85, marker="P", c="dodgerblue", edgecolors="white", linewidths=1.2,
                       zorder=6, label="global optimum")
    else:
        if lo <= 0 <= hi:
            ax.scatter([0], [0], s=85, marker="P", c="dodgerblue", edgecolors="white", linewidths=1.2,
                       zorder=6, label="global optimum")

    # Illustrative local basin centers for regular multimodal functions
    if name.startswith("Rastrigin"):
        # Rastrigin's local minima occur near integer grid points; exact locations are close to them.
        pts = []
        for i in range(int(np.ceil(lo)), int(np.floor(hi)) + 1):
            for j in range(int(np.ceil(lo)), int(np.floor(hi)) + 1):
                if (i, j) != (0, 0) and abs(i) <= 2 and abs(j) <= 2:
                    pts.append((i, j))
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=38, c="crimson", edgecolors="white", linewidths=0.8,
                       zorder=5, label="local basins")
    elif name.startswith("Ackley"):
        # Ackley has many local basins; mark a small neighborhood around the global optimum.
        pts = [(i, j) for i in [-2, -1, 0, 1, 2] for j in [-2, -1, 0, 1, 2] if (i, j) != (0, 0)]
        pts = [(i, j) for i, j in pts if lo <= i <= hi and lo <= j <= hi]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=30, c="crimson", edgecolors="white", linewidths=0.7,
                       zorder=5, label="local basins")

    # Rosenbrock valley guide y=x^2
    if name.startswith("Rosenbrock"):
        xs = np.linspace(max(lo, -2), min(hi, 2), 250)
        ys = xs**2
        mask = (ys >= lo) & (ys <= hi)
        ax.plot(xs[mask], ys[mask], color="white", linewidth=1.2, linestyle="--", alpha=0.9,
                label="valley y=x1²")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right", framealpha=0.85)


def plot_highdim_2d_slice_grid_for_p(p: int, n_pts: int = 260) -> None:
    """
    One figure like the user's example:
      - each panel is a true high-dimensional 2D slice f(x1, x2, fixed x3...xp)
      - rows/columns show all benchmark functions for a fixed p
      - blue marker shows global optimum / global-optimum set
      - red markers show illustrative local basins where locations are regular
    """
    n_fn = len(FUNCTIONS)
    n_cols = 4
    n_rows = int(np.ceil(n_fn / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.6 * n_rows))
    axes = np.asarray(axes).ravel()

    for ax, info in zip(axes, FUNCTIONS):
        lo, hi = info.domain
        X1, X2, Z = make_2d_slice(info, p=p, n_pts=n_pts)

        # Robust color limits make multimodal patterns easier to see when a few edges are very large.
        vmin, vmax = np.nanpercentile(Z, [1, 99])
        contour = ax.contourf(X1, X2, Z, levels=90, cmap=info.cmap, vmin=vmin, vmax=vmax)
        ax.contour(X1, X2, Z, levels=18, colors="k", alpha=0.16, linewidths=0.35)

        add_landscape_markers(ax, info, lo, hi)

        ax.set_title(f"{info.name}\np = {p}", fontsize=11, weight="bold")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15, linestyle=":")
        fig.colorbar(contour, ax=ax, shrink=0.80, label="f(x)")

    for ax in axes[n_fn:]:
        ax.axis("off")

    fig.suptitle(
        f"High-dimensional 2D slices of benchmark functions, p = {p}\n"
        "Vary x1 and x2; fix x3,...,xp at the function-specific slice base",
        fontsize=15,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fname = OUTPUT_DIR / f"HighDim_2D_Slice_Grid_p{p}.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  [HD-2D] Saved -> {fname}")


def plot_highdim_2d_slice_grids() -> None:
    """Create high-dimensional 2D slice grids for p=20,100,200,500."""
    for p in DIMS:
        plot_highdim_2d_slice_grid_for_p(p=p)


# ═══════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("  Benchmark Function Visualizer — high-dimensional 2D slice version")
    print(f"  Output directory: {OUTPUT_DIR.resolve()}")
    print("=" * 72)

    print("\n[1/5] Rendering 3D surfaces...")
    for info in FUNCTIONS:
        print(f"  -> {info.name}")
        plot_3d_surface(info)

    print("\n[2/5] Rendering true 2D contour slices...")
    for info in FUNCTIONS:
        print(f"  -> {info.name}")
        plot_2d_slices(info)

    print("\n[3/5] Rendering 1D cross-sections...")
    for info in FUNCTIONS:
        print(f"  -> {info.name}")
        plot_1d_cross_sections(info)

    print("\n[4/5] Computing and rendering summary table...")
    rows = compute_summary_rows()

    header = f"{'Function':<32} {'p':>5} {'f*':>10} {'sampled min':>14} {'1D local':>10}  Type"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['Function']:<32} {r['p']:>5} {r['Theoretical f*']:>10.4g} "
            f"{r['Sampled 1D min']:>14.6g} {r['Sampled 1D local minima']:>10}  {r['Type']}"
        )

    plot_summary_table(rows)

    print("\n[5/6] Rendering high-dimensional 2D slice grids...")
    plot_highdim_2d_slice_grids()

    print("\n[6/6] Rendering aggregate plots...")
    plot_local_minima_heatmap(rows)
    plot_all_functions_1d_grid()

    print("\n" + "=" * 72)
    print(f"  Done. All plots saved in: {OUTPUT_DIR.resolve()}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
