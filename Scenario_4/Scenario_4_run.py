"""
Scenario 4 — VSR (2D Voxel-based Soft Robot) controller optimization (Section 2.2.4).

Paper uses the 2D-VSR-Sim by Medvet et al. 2020 (Java/JGEA framework). A faithful
re-implementation of that simulator is out of scope. Instead, this file provides
a self-contained, simplified mass-spring 2D simulator that captures the *essence*
of the paper's setup so that the EA pipeline, controllers, sensors, problems, and
metrics match the paper.

What is faithful to the paper:
  - 2 morphologies: Biped (10 voxels) and Tower (14 voxels), shared corners.
  - 3 controller types: C (centralized), HoD (homo-distributed, weight sharing),
    HeD (hetero-distributed, per-voxel ANN).
  - 2 sensor configurations: Homogeneous (area, vx, vy per voxel)
                             Heterogeneous (per-voxel mix incl. proximity, sin)
  - Sensor compatibility constraint:
        Heterogeneous → C, HeD only
        Homogeneous   → HoD only
    => 5 tasks × 3 (controller, sensor) combos = 15 problems.
  - 5 tasks: Locomotion (flat / hilly / steppy), Jump, Balance.
  - Controller invoked every 0.2 s (held constant in between), Δt = 1/60 s.
  - Gaussian sensor noise, variance 0.05.
  - 9 solvers identical to Scenario 1/2/3 (CMA-ES, DE, PSO, ES×3, GA×3).

What is simplified vs. paper:
  - Mass-spring physics is a minimal Verlet integrator (no soft-body friction
    coefficients, no rigid-body collision; ground contact = clamp + friction).
  - Genotype size p depends on input/hidden sizes we choose; numbers won't match
    paper's reported {52, 280, 374, 275, 420} exactly — but the architecture
    family (1 hidden, hidden=input, tanh) and weight-sharing semantics do.
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from dataclasses import dataclass, field
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG (paper defaults)
# ============================================================
N_EVALS_DEFAULT  = 10_000
N_REP_DEFAULT    = 20            # paper: 20 runs for S4 (heavier than S1/2/3)
N_CORES_DEFAULT  = max(1, cpu_count() - 1)
N_SIM_SEEDS_FULL = 1             # one episode per evaluation (paper-style)
N_SIM_SEEDS_QUICK = 1

SOLVER_NAME_MAP = {
    "CMA-ES":  "cmaEs",
    "DE":      "differentialEvolution",
    "PSO":     "pso",
    "ES-0.02": "es-0.02",
    "ES-0.25": "es-0.25",
    "ES-0.5":  "es-0.5",
    "GA-0.02": "ga-0.02",
    "GA-0.25": "ga-0.25",
    "GA-0.5":  "ga-0.5",
}


# ============================================================
# RECORD (same schema as Scenarios 1–3)
# ============================================================
def _record(iteration, total_evals, births, best_fit, t0, pop, fitness,
            n_firsts=1, n_lasts=None):
    n_pop   = len(pop)
    n_lasts = n_lasts if n_lasts is not None else n_pop
    rounded = np.round(pop, 6)
    geno_uni = len(np.unique(rounded, axis=0)) / n_pop
    fit_uni  = len(np.unique(np.round(fitness, 8))) / n_pop
    return {
        "iterations":  iteration,
        "evals":       total_evals,
        "births":      births,
        "elapsed":     round(time.time() - t0, 4),
        "all_size":    n_pop,
        "firsts_size": n_firsts,
        "lasts_size":  n_lasts,
        "geno_uni":    geno_uni,
        "sol_uni":     geno_uni,
        "fit_uni":     fit_uni,
        "best_fitness": float(best_fit),
    }


# ============================================================
# SOLVERS — identical paper-spec versions used in Scenario 2/3
# ============================================================
def run_cma_es(fitness_fn, dim, seed, n_evals: int):
    import cma
    t0  = time.time()
    rng = np.random.default_rng(seed)
    x0  = rng.uniform(-1, 1, dim)
    es  = cma.CMAEvolutionStrategy(
        x0, 0.5, {"seed": int(seed), "verbose": -9, "maxfevals": int(n_evals)}
    )
    records, total_evals, births, iteration = [], 0, 0, 0
    while not es.stop() and total_evals < n_evals:
        solutions = es.ask()
        fits = np.array([fitness_fn(s) for s in solutions])
        es.tell(solutions, fits.tolist())
        n = len(solutions)
        total_evals += n; births += n; iteration += 1
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, births,
                                   float(np.min(fits)), t0,
                                   np.array(solutions), fits, n_lasts=n))
    return records


def run_de(fitness_fn, dim, seed, n_evals: int):
    t0, rng = time.time(), np.random.default_rng(seed)
    NP, F, CR = 15, 0.5, 0.8
    pop     = rng.uniform(-1, 1, (NP, dim))
    fitness = np.array([fitness_fn(ind) for ind in pop])
    total_evals, iteration = NP, 0
    records = [_record(0, NP, NP, np.min(fitness), t0, pop, fitness)]
    while total_evals < n_evals:
        iteration += 1
        for i in range(NP):
            choices    = [j for j in range(NP) if j != i]
            r1, r2, r3 = rng.choice(choices, 3, replace=False)
            mutant     = pop[r1] + F * (pop[r2] - pop[r3])
            mask       = rng.random(dim) < CR
            if not np.any(mask): mask[rng.integers(dim)] = True
            trial      = np.where(mask, mutant, pop[i])
            f_trial    = fitness_fn(trial); total_evals += 1
            if f_trial <= fitness[i]:
                pop[i], fitness[i] = trial, f_trial
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, total_evals,
                                   np.min(fitness), t0, pop, fitness))
    return records


def run_pso(fitness_fn, dim, seed, n_evals: int):
    t0, rng = time.time(), np.random.default_rng(seed)
    n_pop, w, phi_p, phi_g = 100, 0.8, 1.5, 1.5
    pos = rng.uniform(-1, 1, (n_pop, dim))
    vel = rng.uniform(-0.5, 0.5, (n_pop, dim))
    fit = np.array([fitness_fn(p) for p in pos])
    pbest_pos, pbest_fit = pos.copy(), fit.copy()
    gbest_idx = np.argmin(fit)
    gbest_fit, gbest_pos = fit[gbest_idx], pos[gbest_idx].copy()
    total_evals, iteration = n_pop, 0
    records = [_record(0, n_pop, n_pop, gbest_fit, t0, pos, fit)]
    while total_evals < n_evals:
        iteration += 1
        r1, r2 = rng.random((n_pop, dim)), rng.random((n_pop, dim))
        vel = w * vel + phi_p * r1 * (pbest_pos - pos) + phi_g * r2 * (gbest_pos - pos)
        pos = pos + vel
        fit = np.array([fitness_fn(p) for p in pos])
        total_evals += n_pop
        improved = fit < pbest_fit
        pbest_pos[improved] = pos[improved]; pbest_fit[improved] = fit[improved]
        idx = np.argmin(pbest_fit)
        if pbest_fit[idx] < gbest_fit:
            gbest_fit = pbest_fit[idx]; gbest_pos = pbest_pos[idx].copy()
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, total_evals,
                                   gbest_fit, t0, pos, fit))
    return records


def run_es(fitness_fn, dim, seed, sigma: float, n_evals: int):
    t0, rng = time.time(), np.random.default_rng(seed)
    n_pop     = 30
    n_parents = int(np.floor(0.33 * n_pop))
    pop = rng.uniform(-1, 1, (n_pop, dim))
    fit = np.array([fitness_fn(ind) for ind in pop])
    total_evals, births, iteration = n_pop, n_pop, 0
    records = [_record(0, n_pop, n_pop, np.min(fit), t0, pop, fit)]
    std = float(np.sqrt(sigma))
    while total_evals < n_evals:
        iteration += 1
        order   = np.argsort(fit)
        mu_vec  = pop[order[:n_parents]].mean(axis=0)
        new_pop = np.empty_like(pop)
        new_pop[0]  = pop[order[0]]
        new_pop[1:] = mu_vec + rng.normal(0.0, std, size=(n_pop - 1, dim))
        pop = new_pop
        fit = np.array([fitness_fn(ind) for ind in pop])
        total_evals += n_pop; births += n_pop
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, births,
                                   np.min(fit), t0, pop, fit, n_lasts=n_pop))
    return records


def run_ga(fitness_fn, dim, seed, sigma: float, n_evals: int):
    t0, rng = time.time(), np.random.default_rng(seed)
    n_pop, n_tour, p_xo = 100, 5, 0.8
    pop = rng.uniform(-1, 1, (n_pop, dim))
    fit = np.array([fitness_fn(ind) for ind in pop])
    total_evals, births, iteration = n_pop, n_pop, 0
    records = [_record(0, n_pop, n_pop, np.min(fit), t0, pop, fit)]
    std = float(np.sqrt(sigma))

    def tour_select():
        parts = rng.choice(n_pop, n_tour, replace=False)
        return pop[parts[np.argmin(fit[parts])]]

    while total_evals < n_evals:
        iteration += 1
        offspring = np.empty_like(pop)
        for k in range(n_pop):
            if rng.random() < p_xo:
                x1, x2 = tour_select(), tour_select()
                alpha  = rng.random()
                child  = x1 + alpha * (x2 - x1) + rng.normal(0.0, std, size=dim)
            else:
                child  = tour_select() + rng.normal(0.0, std, size=dim)
            offspring[k] = child
        off_fit = np.array([fitness_fn(o) for o in offspring])
        births += n_pop; total_evals += n_pop
        merged_pop = np.vstack([pop, offspring])
        merged_fit = np.concatenate([fit, off_fit])
        best_idx   = np.argsort(merged_fit)[:n_pop]
        pop, fit   = merged_pop[best_idx], merged_fit[best_idx]
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, births,
                                   np.min(fit), t0, pop, fit, n_lasts=n_pop))
    return records


# ============================================================
# VSR PHYSICS (simplified mass-spring, paper-inspired)
# ============================================================
DT_SIM         = 1.0 / 60.0   # 60 Hz physics step (paper)
CONTROL_PERIOD = 0.2          # 0.2 s controller invocation period (paper)
GRAVITY        = -9.8
VOXEL_SIZE     = 1.0          # 1 m default (abstract units)
SPRING_K       = 80.0         # spring stiffness
SPRING_K_DIAG  = 50.0         # diagonal stiffness (slightly weaker)
DAMPING        = 0.5          # spring damping
GROUND_FRIC    = 0.85         # tangential friction factor on ground contact
MASS           = 1.0          # mass per corner node
CONTRACT_RANGE = 0.3          # ±30% rest-length modulation
SENSOR_NOISE_VAR = 0.05       # paper

# Morphology grids: (cols, rows). Cell value 1 = voxel present.
def _biped_grid() -> np.ndarray:
    """Biped: 4×3 grid minus 2 top corners ⇒ 10 voxels."""
    g = np.ones((4, 3), dtype=int)   # cols × rows
    g[0, 2] = 0
    g[3, 2] = 0
    return g

def _tower_grid() -> np.ndarray:
    """Tower: 2×7 grid ⇒ 14 voxels (vertical column)."""
    return np.ones((2, 7), dtype=int)


@dataclass
class VSRBody:
    """Mass-spring representation built from a voxel grid."""
    nodes: np.ndarray           # (N, 2) positions
    velocities: np.ndarray      # (N, 2)
    springs: np.ndarray         # (M, 2) node-index pairs
    rest_lengths: np.ndarray    # (M,)
    spring_k: np.ndarray        # (M,)
    voxel_corners: np.ndarray   # (V, 4) node indices per voxel (BL, BR, TR, TL)
    voxel_springs: np.ndarray   # (V, 6) spring indices per voxel
    initial_area: np.ndarray    # (V,)
    n_voxels: int
    n_nodes: int


def build_vsr(grid: np.ndarray, voxel_size: float = VOXEL_SIZE,
              y_offset: float = 0.5) -> VSRBody:
    """Build a VSRBody from a 2D voxel grid using shared-corner mass-spring."""
    cols, rows = grid.shape
    node_idx: Dict[Tuple[int, int], int] = {}
    nodes: List[List[float]] = []

    # Allocate nodes only at corners that belong to a present voxel
    for c in range(cols):
        for r in range(rows):
            if grid[c, r] == 0:
                continue
            for dx, dy in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                key = (c + dx, r + dy)
                if key not in node_idx:
                    node_idx[key] = len(nodes)
                    nodes.append([key[0] * voxel_size,
                                  key[1] * voxel_size + y_offset])

    nodes_arr  = np.array(nodes, dtype=np.float64)
    velocities = np.zeros_like(nodes_arr)

    # Springs: 4 edges + 2 diagonals per voxel
    spring_set: Dict[Tuple[int, int], Tuple[float, float]] = {}
    voxel_corners: List[List[int]] = []
    voxel_springs: List[List[int]] = []
    initial_area: List[float] = []

    def _spring(a: int, b: int, k: float) -> int:
        key = (min(a, b), max(a, b))
        if key not in spring_set:
            rest = float(np.linalg.norm(nodes_arr[a] - nodes_arr[b]))
            spring_set[key] = (rest, k)
        return list(spring_set.keys()).index(key)

    spring_keys: List[Tuple[int, int]] = []
    spring_data: List[Tuple[float, float]] = []

    def _ensure_spring(a: int, b: int, k: float) -> int:
        key = (min(a, b), max(a, b))
        for i, sk in enumerate(spring_keys):
            if sk == key:
                return i
        rest = float(np.linalg.norm(nodes_arr[a] - nodes_arr[b]))
        spring_keys.append(key)
        spring_data.append((rest, k))
        return len(spring_keys) - 1

    for c in range(cols):
        for r in range(rows):
            if grid[c, r] == 0:
                continue
            bl = node_idx[(c, r)]
            br = node_idx[(c + 1, r)]
            tr = node_idx[(c + 1, r + 1)]
            tl = node_idx[(c, r + 1)]
            corners = [bl, br, tr, tl]
            voxel_corners.append(corners)

            sids = [
                _ensure_spring(bl, br, SPRING_K),       # bottom
                _ensure_spring(br, tr, SPRING_K),       # right
                _ensure_spring(tr, tl, SPRING_K),       # top
                _ensure_spring(tl, bl, SPRING_K),       # left
                _ensure_spring(bl, tr, SPRING_K_DIAG),  # diag /
                _ensure_spring(br, tl, SPRING_K_DIAG),  # diag \
            ]
            voxel_springs.append(sids)
            initial_area.append(voxel_size * voxel_size)

    springs = np.array([list(k) for k in spring_keys], dtype=np.int32)
    rest_lengths = np.array([d[0] for d in spring_data], dtype=np.float64)
    spring_k = np.array([d[1] for d in spring_data], dtype=np.float64)

    return VSRBody(
        nodes=nodes_arr,
        velocities=velocities,
        springs=springs,
        rest_lengths=rest_lengths,
        spring_k=spring_k,
        voxel_corners=np.array(voxel_corners, dtype=np.int32),
        voxel_springs=np.array(voxel_springs, dtype=np.int32),
        initial_area=np.array(initial_area, dtype=np.float64),
        n_voxels=len(voxel_corners),
        n_nodes=len(nodes_arr),
    )


def _voxel_centers(body: VSRBody) -> np.ndarray:
    """Return (V, 2) center of each voxel."""
    return body.nodes[body.voxel_corners].mean(axis=1)


def _voxel_areas(body: VSRBody) -> np.ndarray:
    """Compute current area of each voxel via shoelace on its 4 corners."""
    pts = body.nodes[body.voxel_corners]                # (V, 4, 2)
    x = pts[:, :, 0]
    y = pts[:, :, 1]
    # Shoelace
    return 0.5 * np.abs(
        x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
        + x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1]
        + x[:, 2] * y[:, 3] - x[:, 3] * y[:, 2]
        + x[:, 3] * y[:, 0] - x[:, 0] * y[:, 3]
    )


def _voxel_velocities(body: VSRBody) -> np.ndarray:
    return body.velocities[body.voxel_corners].mean(axis=1)   # (V, 2)


def physics_step(body: VSRBody, terrain_fn, contractions: np.ndarray,
                 dt: float = DT_SIM):
    """One Verlet-style mass-spring step with gravity + ground."""
    # Apply contraction: scale rest_lengths of each voxel's 6 springs.
    # Vectorized: spring i may belong to multiple voxels → take mean scale.
    scale_per_voxel = 1.0 + CONTRACT_RANGE * contractions          # (V,)
    spring_scale_sum = np.ones(body.rest_lengths.shape[0])
    spring_scale_cnt = np.ones(body.rest_lengths.shape[0])         # avoid /0
    flat_sids   = body.voxel_springs.ravel()                       # (V*6,)
    flat_scales = np.repeat(scale_per_voxel, body.voxel_springs.shape[1])
    np.add.at(spring_scale_sum, flat_sids, flat_scales)
    np.add.at(spring_scale_cnt, flat_sids, np.ones_like(flat_sids, dtype=float))
    avg_scale = spring_scale_sum / spring_scale_cnt
    rest_modulated = body.rest_lengths * avg_scale

    # Compute spring forces
    a = body.springs[:, 0]
    b = body.springs[:, 1]
    pa = body.nodes[a]
    pb = body.nodes[b]
    delta = pb - pa
    length = np.linalg.norm(delta, axis=1) + 1e-9
    direction = delta / length[:, None]
    rel_vel = body.velocities[b] - body.velocities[a]
    rel_vel_along = (rel_vel * direction).sum(axis=1)
    spring_force_mag = body.spring_k * (length - rest_modulated)
    damping_force_mag = DAMPING * rel_vel_along
    f_mag = spring_force_mag + damping_force_mag
    f_vec = direction * f_mag[:, None]                         # (M, 2)

    # Accumulate forces on nodes
    forces = np.zeros_like(body.nodes)
    np.add.at(forces, a, f_vec)
    np.subtract.at(forces, b, f_vec)

    # Gravity
    forces[:, 1] += MASS * GRAVITY

    # Integrate
    accel = forces / MASS
    body.velocities += accel * dt
    body.nodes      += body.velocities * dt

    # Vectorized ground collision (terrain_fn must be elementwise-safe)
    ground_ys = np.array([terrain_fn(x) for x in body.nodes[:, 0]])
    below     = body.nodes[:, 1] < ground_ys
    if below.any():
        body.nodes[below, 1] = ground_ys[below]
        # Zero downward vy and apply tangential friction
        vy_neg = body.velocities[below, 1] < 0
        idx_below = np.where(below)[0]
        body.velocities[idx_below[vy_neg], 1] = 0.0
        body.velocities[below, 0] *= GROUND_FRIC


# ============================================================
# SENSORS
# ============================================================
def _make_sensor_layout(n_voxels: int, sensor_config: str) -> List[int]:
    """
    Returns a list of length n_voxels where element i = number of sensors of voxel i.
    Homogeneous: every voxel has 3 sensors (area, vx, vy).
    Heterogeneous: voxels split into groups with different sensor counts:
        head/top group   : 4 sensors  (area, vx, vy, sin)
        body/middle      : 3 sensors  (area, vx, vy)
        bottom/feet      : 5 sensors  (area, vx, vy, proximity, sin)
    """
    if sensor_config == "homo":
        return [3] * n_voxels
    # Heterogeneous: split into thirds (top/middle/bottom)
    layout = []
    third = max(1, n_voxels // 3)
    for i in range(n_voxels):
        if i < third:
            layout.append(5)   # bottom
        elif i < 2 * third:
            layout.append(3)   # middle
        else:
            layout.append(4)   # top
    return layout


def read_sensors(body: VSRBody, sensor_layout: List[int],
                 t: float, rng: np.random.Generator) -> List[np.ndarray]:
    """Return list of per-voxel sensor arrays (each np.ndarray)."""
    areas = _voxel_areas(body) / body.initial_area     # area ratio
    vels  = _voxel_velocities(body)                    # (V, 2)
    centers = _voxel_centers(body)                     # (V, 2)
    sin_signal = math.sin(2.0 * math.pi * 1.0 * t)     # 1 Hz

    out: List[np.ndarray] = []
    noise_std = math.sqrt(SENSOR_NOISE_VAR)
    for i, n_s in enumerate(sensor_layout):
        s = []
        s.append(np.clip(areas[i] - 1.0, -1.0, 1.0))   # area-ratio (centered)
        s.append(np.clip(vels[i, 0] / 5.0, -1.0, 1.0)) # vx normalized
        s.append(np.clip(vels[i, 1] / 5.0, -1.0, 1.0)) # vy normalized
        if n_s >= 4:
            s.append(np.clip(sin_signal, -1.0, 1.0))
        if n_s >= 5:
            # crude proximity: distance to ground at voxel center, clipped
            ground_y = 0.0  # ground baseline
            prox = np.clip((centers[i, 1] - ground_y) / 2.0, -1.0, 1.0)
            s.append(prox)
        arr = np.array(s, dtype=np.float64)
        arr = arr + rng.normal(0.0, noise_std, size=arr.shape)
        out.append(arr)
    return out


# ============================================================
# CONTROLLERS  (C / HoD / HeD)
# ============================================================
@dataclass(frozen=True)
class ControllerConfig:
    controller_type: str  # 'C' | 'HoD' | 'HeD'
    sensor_config: str    # 'homo' | 'hetero'
    n_voxels: int
    sensor_layout: Tuple[int, ...]  # per-voxel sensor count

    @property
    def total_sensors(self) -> int:
        return sum(self.sensor_layout)


def theta_dim(cfg: ControllerConfig) -> int:
    """Compute number of ANN parameters for the given controller config."""
    if cfg.controller_type == "C":
        n_in  = cfg.total_sensors
        n_hid = n_in
        n_out = cfg.n_voxels
        return n_in * n_hid + n_hid + n_hid * n_out + n_out

    if cfg.controller_type == "HoD":
        # Single shared ANN per voxel (sensor layout must be homogeneous)
        assert cfg.sensor_config == "homo", "HoD requires homogeneous sensors"
        n_in  = cfg.sensor_layout[0]
        n_hid = n_in
        n_out = 1
        return n_in * n_hid + n_hid + n_hid * n_out + n_out

    if cfg.controller_type == "HeD":
        total = 0
        for n_s in cfg.sensor_layout:
            n_in  = n_s
            n_hid = n_in
            n_out = 1
            total += n_in * n_hid + n_hid + n_hid * n_out + n_out
        return total

    raise ValueError(f"Unknown controller_type {cfg.controller_type}")


def predict_contractions(theta: np.ndarray, sensors: List[np.ndarray],
                         cfg: ControllerConfig) -> np.ndarray:
    """Returns (n_voxels,) contraction values in [-1, 1] from a flat θ."""
    if cfg.controller_type == "C":
        x = np.concatenate(sensors)
        n_in = cfg.total_sensors; n_hid = n_in; n_out = cfg.n_voxels
        idx = 0
        W1 = theta[idx: idx + n_in * n_hid].reshape(n_in, n_hid); idx += n_in * n_hid
        b1 = theta[idx: idx + n_hid];                              idx += n_hid
        W2 = theta[idx: idx + n_hid * n_out].reshape(n_hid, n_out); idx += n_hid * n_out
        b2 = theta[idx: idx + n_out]
        h  = np.tanh(x @ W1 + b1)
        return np.tanh(h @ W2 + b2)

    if cfg.controller_type == "HoD":
        n_in  = cfg.sensor_layout[0]; n_hid = n_in; n_out = 1
        idx = 0
        W1 = theta[idx: idx + n_in * n_hid].reshape(n_in, n_hid); idx += n_in * n_hid
        b1 = theta[idx: idx + n_hid];                              idx += n_hid
        W2 = theta[idx: idx + n_hid * n_out].reshape(n_hid, n_out); idx += n_hid * n_out
        b2 = theta[idx: idx + n_out]
        out = np.empty(cfg.n_voxels)
        for i in range(cfg.n_voxels):
            h = np.tanh(sensors[i] @ W1 + b1)
            out[i] = float(np.tanh(h @ W2 + b2)[0])
        return out

    # HeD
    out = np.empty(cfg.n_voxels)
    idx = 0
    for i, n_s in enumerate(cfg.sensor_layout):
        n_in = n_s; n_hid = n_in; n_out_v = 1
        W1 = theta[idx: idx + n_in * n_hid].reshape(n_in, n_hid); idx += n_in * n_hid
        b1 = theta[idx: idx + n_hid];                              idx += n_hid
        W2 = theta[idx: idx + n_hid * n_out_v].reshape(n_hid, n_out_v); idx += n_hid * n_out_v
        b2 = theta[idx: idx + n_out_v];                            idx += n_out_v
        h  = np.tanh(sensors[i] @ W1 + b1)
        out[i] = float(np.tanh(h @ W2 + b2)[0])
    return out


# ============================================================
# TASKS (5 fitness functions)
# ============================================================
def _terrain_flat(_x: float) -> float:
    return 0.0

def _terrain_hilly(x: float) -> float:
    return 0.15 * math.sin(0.6 * x)

def _terrain_steppy(x: float) -> float:
    # 0.1 m step every 1 m
    return 0.10 * math.floor(x)


def _simulate_episode(theta: np.ndarray, body: VSRBody, cfg: ControllerConfig,
                      terrain_fn, t_total: float, rng: np.random.Generator,
                      record_traj: bool = False):
    """
    Run one episode. Returns dict with diagnostics.
    Controller invoked every CONTROL_PERIOD; physics at DT_SIM.
    """
    steps_total = int(t_total / DT_SIM)
    control_every = max(1, int(round(CONTROL_PERIOD / DT_SIM)))

    contractions = np.zeros(body.n_voxels)
    com_traj_x = np.zeros(steps_total)
    com_traj_y = np.zeros(steps_total)

    for s in range(steps_total):
        t = s * DT_SIM
        if s % control_every == 0:
            sensors = read_sensors(body, list(cfg.sensor_layout), t, rng)
            contractions = predict_contractions(theta, sensors, cfg)
        physics_step(body, terrain_fn, contractions)
        com = body.nodes.mean(axis=0)
        com_traj_x[s] = com[0]
        com_traj_y[s] = com[1]

    return {
        "com_x": com_traj_x,
        "com_y": com_traj_y,
        "final_com": body.nodes.mean(axis=0),
        "initial_com_x": com_traj_x[0],
    }


# ============================================================
# PROBLEM SPEC
# ============================================================
@dataclass(frozen=True)
class ProblemSpec:
    task: str             # 'flat'|'hilly'|'steppy'|'jump'|'balance'
    morphology: str       # 'biped' | 'tower'
    controller_type: str  # 'C' | 'HoD' | 'HeD'
    sensor_config: str    # 'homo' | 'hetero'
    objective: str        # 'maximize' | 'minimize'
    problem_name: str


def build_problem_specs() -> List[ProblemSpec]:
    """
    15 problems = 5 tasks × 3 controller-sensor combos.
    Combos (per paper): (HoD, homo), (C, hetero), (HeD, hetero).
    Tasks 1–4 use Biped; task 5 (Balance) uses Tower.
    """
    specs: List[ProblemSpec] = []
    combos = [
        ("HoD", "homo"),
        ("C",   "hetero"),
        ("HeD", "hetero"),
    ]
    tasks_biped = [
        ("flat",  "maximize"),
        ("hilly", "maximize"),
        ("steppy", "maximize"),
        ("jump",  "maximize"),
    ]
    for task, obj in tasks_biped:
        for ctrl, sens in combos:
            specs.append(ProblemSpec(
                task=task,
                morphology="biped",
                controller_type=ctrl,
                sensor_config=sens,
                objective=obj,
                problem_name=f"ea.p.v.{task}-{ctrl}",
            ))
    for ctrl, sens in combos:
        specs.append(ProblemSpec(
            task="balance",
            morphology="tower",
            controller_type=ctrl,
            sensor_config=sens,
            objective="minimize",
            problem_name=f"ea.p.v.balance-{ctrl}",
        ))
    return specs


def build_controller_cfg(spec: ProblemSpec) -> ControllerConfig:
    n_voxels = 10 if spec.morphology == "biped" else 14
    layout = _make_sensor_layout(n_voxels, spec.sensor_config)
    return ControllerConfig(
        controller_type=spec.controller_type,
        sensor_config=spec.sensor_config,
        n_voxels=n_voxels,
        sensor_layout=tuple(layout),
    )


# ============================================================
# FITNESS WRAPPING (paper-style, all minimize internally)
# ============================================================
def make_fitness(spec: ProblemSpec, t_sim: float):
    """
    Returns a fitness_fn(theta) that returns a number to MINIMIZE.
    For maximize-tasks we negate the metric so the EA can minimize uniformly.
    """
    cfg = build_controller_cfg(spec)
    grid = _biped_grid() if spec.morphology == "biped" else _tower_grid()

    if spec.task == "flat":
        terrain = _terrain_flat
    elif spec.task == "hilly":
        terrain = _terrain_hilly
    elif spec.task == "steppy":
        terrain = _terrain_steppy
    elif spec.task == "jump":
        terrain = _terrain_flat
    elif spec.task == "balance":
        terrain = _terrain_flat
    else:
        raise ValueError(f"Unknown task {spec.task}")

    def fitness_fn(theta: np.ndarray) -> float:
        rng = np.random.default_rng(0)            # deterministic sensor noise per eval
        body = build_vsr(grid)
        # Settle a tiny bit so initial spring tension dissipates
        for _ in range(5):
            physics_step(body, terrain, np.zeros(body.n_voxels))

        if spec.task in ("flat", "hilly", "steppy"):
            traj = _simulate_episode(theta, body, cfg, terrain, t_total=min(t_sim, 30.0), rng=rng)
            # mean horizontal velocity = (final_x - initial_x) / time
            displacement = traj["final_com"][0] - traj["initial_com_x"]
            mean_vx = displacement / max(t_sim, 1e-6)
            return -float(mean_vx)                # minimize negative ⇒ maximize vx

        if spec.task == "jump":
            traj = _simulate_episode(theta, body, cfg, terrain, t_total=min(t_sim, 10.0), rng=rng)
            steps = len(traj["com_y"])
            warmup = int(steps * 0.5)             # ignore first 5s of 10s episode
            max_h = float(np.max(traj["com_y"][warmup:]) - traj["com_y"][0])
            return -max_h                         # maximize jump height

        if spec.task == "balance":
            traj = _simulate_episode(theta, body, cfg, terrain, t_total=min(t_sim, 30.0), rng=rng)
            # angle proxy: average horizontal drift of COM from initial position
            drift = float(np.mean(np.abs(traj["com_x"] - traj["initial_com_x"])))
            return drift                          # minimize drift

        raise ValueError(f"Unknown task {spec.task}")

    return fitness_fn


# ============================================================
# WORKER
# ============================================================
def _worker(args):
    spec, solver_key, sigma, seed, n_evals, t_sim = args

    cfg = build_controller_cfg(spec)
    dim = theta_dim(cfg)
    fn  = make_fitness(spec, t_sim)

    if solver_key == "CMA-ES":
        records = run_cma_es(fn, dim, seed, n_evals)
    elif solver_key == "DE":
        records = run_de(fn, dim, seed, n_evals)
    elif solver_key == "PSO":
        records = run_pso(fn, dim, seed, n_evals)
    elif solver_key.startswith("ES-"):
        records = run_es(fn, dim, seed, float(sigma), n_evals)
    elif solver_key.startswith("GA-"):
        records = run_ga(fn, dim, seed, float(sigma), n_evals)
    else:
        records = []

    j_solv = SOLVER_NAME_MAP[solver_key]
    return [
        {
            **r,
            "seed":          seed + 1,
            "problem":       spec.problem_name,
            "solver_sigma":  j_solv,
            "objective":     spec.objective,
            "best→fitness":  r["best_fitness"],
            "genotype_size": dim,
        }
        for r in records
    ]


# ============================================================
# MAIN
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scenario 4 — VSR controller optimization")
    p.add_argument("--n_evals", type=int, default=N_EVALS_DEFAULT)
    p.add_argument("--n_rep",   type=int, default=N_REP_DEFAULT)
    p.add_argument("--cores",   type=int, default=N_CORES_DEFAULT)
    p.add_argument("--quick",   action="store_true",
                   help="Smoke test: 2 reps, 300 evals, short sim, 1 problem.")
    p.add_argument("--problems", nargs="*", default=None,
                   help="Subset of problem names, e.g. ea.p.v.flat-HoD")
    p.add_argument("--t_sim", type=float, default=None,
                   help="Override simulation duration (s) per episode.")
    return p.parse_args()


def main():
    args     = parse_args()
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.n_rep   = min(args.n_rep, 2)
        args.n_evals = min(args.n_evals, 300)
        args.cores   = 1
        t_sim = args.t_sim if args.t_sim is not None else 3.0
    else:
        t_sim = args.t_sim if args.t_sim is not None else 30.0

    specs = build_problem_specs()
    if args.problems:
        wanted = set(args.problems)
        specs = [s for s in specs if s.problem_name in wanted]
        if not specs:
            raise ValueError("No matching problems. Example: ea.p.v.flat-HoD")

    # Build task list
    tasks = []
    for spec in specs:
        for sk in SOLVER_NAME_MAP:
            sig = float(sk.split("-")[1]) if sk.startswith("ES-") or sk.startswith("GA-") else None
            for s in range(args.n_rep):
                tasks.append((spec, sk, sig, s, args.n_evals, t_sim))

    all_rows: list = []

    if args.cores <= 1:
        for t in tqdm(tasks, total=len(tasks)):
            all_rows.extend(_worker(t))
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=args.cores) as pool:
            for res in tqdm(
                pool.imap_unordered(_worker, tasks, chunksize=1),
                total=len(tasks),
            ):
                all_rows.extend(res)

    out_csv = results_dir / "Scenario_4_Novel_Final.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, sep=";", index=False)
    print(f"Done! Saved to {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
