import argparse
import os
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from tqdm import tqdm

from multiprocessing import cpu_count, get_context

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG (paper defaults)
# ============================================================
N_EVALS_DEFAULT = 10_000
N_REP_DEFAULT = 30
N_CORES_DEFAULT = max(1, cpu_count() - 1)

SOLVER_NAME_MAP = {
    "CMA-ES": "cmaEs",
    "DE": "differentialEvolution",
    "PSO": "pso",
    "ES-0.02": "es-0.02",
    "ES-0.25": "es-0.25",
    "ES-0.5": "es-0.5",
    "GA-0.02": "ga-0.02",
    "GA-0.25": "ga-0.25",
    "GA-0.5": "ga-0.5",
}

# ============================================================
# RECORD (same schema as Scenario 1)
# ============================================================
def _record(iteration, total_evals, births, best_fit, t0, pop, fitness, n_firsts=1, n_lasts=None):
    n_pop = len(pop)
    n_lasts = n_lasts if n_lasts is not None else n_pop

    rounded_pop = np.round(pop, 6)
    geno_uni = len(np.unique(rounded_pop, axis=0)) / n_pop
    fit_uni = len(np.unique(np.round(fitness, 8))) / n_pop

    return {
        "iterations": iteration,
        "evals": total_evals,
        "births": births,
        "elapsed": round(time.time() - t0, 4),
        "all_size": n_pop,
        "firsts_size": n_firsts,
        "lasts_size": n_lasts,
        "geno_uni": geno_uni,
        "sol_uni": geno_uni,
        "fit_uni": fit_uni,
        "best_fitness": float(best_fit),
    }


# ============================================================
# SOLVERS (match paper / tai_lieu/algorithms.md)
# - Init: U([-1, 1])^p for all
# - Termination: stop at first iteration where evals would exceed n_evals
# - Objective: minimize
# ============================================================
def run_cma_es(fitness_fn, dim, seed, n_evals: int):
    import cma

    t0 = time.time()

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1, 1, dim)

    es = cma.CMAEvolutionStrategy(
        x0,
        0.5,
        {"seed": int(seed), "verbose": -9, "maxfevals": int(n_evals)},
    )

    records = []
    total_evals = 0
    births = 0
    iteration = 0

    while not es.stop() and total_evals < n_evals:
        solutions = es.ask()
        fits = np.array([fitness_fn(s) for s in solutions])
        es.tell(solutions, fits.tolist())

        n = len(solutions)
        total_evals += n
        births += n
        iteration += 1

        best_so_far = float(es.best.f)

        if iteration % 5 == 0 or total_evals >= n_evals:
            records.append(
                _record(
                    iteration,
                    total_evals,
                    births,
                    best_so_far,
                    t0,
                    np.array(solutions),
                    fits,
                    n_lasts=n,
                )
            )
    return records


def run_de(fitness_fn, dim, seed, n_evals: int):
    t0, rng = time.time(), np.random.default_rng(seed)

    # Paper: DE/rand/1/bin with NP=15, F=0.5, CR=0.8
    NP, F, CR = 15, 0.5, 0.8

    pop = rng.uniform(-1, 1, (NP, dim))
    fitness = np.array([fitness_fn(ind) for ind in pop])

    total_evals, iteration = NP, 0
    records = [_record(0, NP, NP, np.min(fitness), t0, pop, fitness)]

    while total_evals < n_evals:
        iteration += 1

        for i in range(NP):
            # pick distinct r1,r2,r3 all != i
            choices = [j for j in range(NP) if j != i]
            r1, r2, r3 = rng.choice(choices, 3, replace=False)

            mutant = pop[r1] + F * (pop[r2] - pop[r3])

            mask = rng.random(dim) < CR
            if not np.any(mask):
                mask[rng.integers(dim)] = True

            trial = np.where(mask, mutant, pop[i])

            f_trial = fitness_fn(trial)
            total_evals += 1

            if f_trial <= fitness[i]:
                pop[i], fitness[i] = trial, f_trial

        if iteration % 5 == 0 or total_evals >= n_evals:
            records.append(_record(iteration, total_evals, total_evals, np.min(fitness), t0, pop, fitness))
    return records


def run_pso(fitness_fn, dim, seed, n_evals: int):
    t0, rng = time.time(), np.random.default_rng(seed)

    # Paper (Clerc 2012 params):
    # n_pop=100, inertia w=0.8, phi_particle=1.5, phi_global=1.5
    n_pop = 100
    w = 0.8
    phi_particle = 1.5
    phi_global = 1.5

    pos = rng.uniform(-1, 1, (n_pop, dim))
    vel = rng.uniform(-0.5, 0.5, (n_pop, dim))

    fit = np.array([fitness_fn(p) for p in pos])

    pbest_pos = pos.copy()
    pbest_fit = fit.copy()

    gbest_idx = np.argmin(fit)
    gbest_fit = fit[gbest_idx]
    gbest_pos = pos[gbest_idx].copy()

    total_evals, iteration = n_pop, 0
    records = [_record(0, n_pop, n_pop, gbest_fit, t0, pos, fit)]

    while total_evals < n_evals:
        iteration += 1

        r1 = rng.random((n_pop, dim))
        r2 = rng.random((n_pop, dim))

        vel = w * vel + phi_particle * r1 * (pbest_pos - pos) + phi_global * r2 * (gbest_pos - pos)
        pos = pos + vel

        fit = np.array([fitness_fn(p) for p in pos])
        total_evals += n_pop

        improved = fit < pbest_fit
        pbest_pos[improved] = pos[improved]
        pbest_fit[improved] = fit[improved]

        idx = np.argmin(pbest_fit)
        if pbest_fit[idx] < gbest_fit:
            gbest_fit = pbest_fit[idx]
            gbest_pos = pbest_pos[idx].copy()

        if iteration % 5 == 0 or total_evals >= n_evals:
            records.append(_record(iteration, total_evals, total_evals, gbest_fit, t0, pos, fit))
    return records


def run_es(fitness_fn, dim, seed, sigma: float, n_evals: int):
    t0, rng = time.time(), np.random.default_rng(seed)

    # Paper basic ES:
    # n_pop=30, n_parents=floor(0.33*n_pop)=10, elitism keep best
    n_pop = 30
    n_parents = int(np.floor(0.33 * n_pop))

    pop = rng.uniform(-1, 1, (n_pop, dim))
    fit = np.array([fitness_fn(ind) for ind in pop])

    total_evals, births, iteration = n_pop, n_pop, 0
    records = [_record(0, n_pop, n_pop, np.min(fit), t0, pop, fit)]

    while total_evals < n_evals:
        iteration += 1

        order = np.argsort(fit)
        parents = pop[order[:n_parents]]
        mu = parents.mean(axis=0)

        # elitism: keep best
        new_pop = np.empty_like(pop)
        new_pop[0] = pop[order[0]]

        # Paper labels the variants by Gaussian step-size sigma.
        new_pop[1:] = mu + rng.normal(0.0, sigma, size=(n_pop - 1, dim))

        pop = new_pop
        fit = np.array([fitness_fn(ind) for ind in pop])
        total_evals += n_pop
        births += n_pop

        if iteration % 5 == 0 or total_evals >= n_evals:
            records.append(_record(iteration, total_evals, births, np.min(fit), t0, pop, fit, n_lasts=n_pop))

    return records


def run_ga(fitness_fn, dim, seed, sigma: float, n_evals: int):
    t0, rng = time.time(), np.random.default_rng(seed)

    # Paper GA (continuous):
    # n_pop=100, p_xo=0.8, n_tour=5
    n_pop, n_tour, p_xo = 100, 5, 0.8

    pop = rng.uniform(-1, 1, (n_pop, dim))
    fit = np.array([fitness_fn(ind) for ind in pop])

    total_evals, births, iteration = n_pop, n_pop, 0
    records = [_record(0, n_pop, n_pop, np.min(fit), t0, pop, fit)]

    def tour_select() -> np.ndarray:
        parts = rng.choice(n_pop, n_tour, replace=False)
        return pop[parts[np.argmin(fit[parts])]]

    while total_evals < n_evals:
        iteration += 1
        offspring = np.empty_like(pop)

        for k in range(n_pop):
            if rng.random() < p_xo:
                x1 = tour_select()
                x2 = tour_select()
                alpha = rng.random()
                eps = rng.normal(0.0, sigma, size=dim)
                child = x1 + alpha * (x2 - x1) + eps
            else:
                x = tour_select()
                eps = rng.normal(0.0, sigma, size=dim)
                child = x + eps
            offspring[k] = child

        off_fit = np.array([fitness_fn(o) for o in offspring])

        births += n_pop
        total_evals += n_pop

        merged_pop = np.vstack([pop, offspring])
        merged_fit = np.concatenate([fit, off_fit])
        best_idx = np.argsort(merged_fit)[:n_pop]
        pop = merged_pop[best_idx]
        fit = merged_fit[best_idx]

        if iteration % 5 == 0 or total_evals >= n_evals:
            records.append(_record(iteration, total_evals, births, np.min(fit), t0, pop, fit, n_lasts=n_pop))

    return records


# ============================================================
# DATA: download + preprocess (per your choices)
# ============================================================
UCI_CONCRETE_XLS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
UCI_ENERGY_ZIP_URL = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
UCI_WINE_WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
UCI_ABALONE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
UCI_AUTO_MPG_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"


def _download_if_needed(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    urlretrieve(url, dst.as_posix())
    return dst


def _standardize_x(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mu) / std


def _rescale_y_to_minus1_1(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float64)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if y_max == y_min:
        return np.zeros_like(y, dtype=np.float64)
    y01 = (y - y_min) / (y_max - y_min)
    return y01 * 2.0 - 1.0


def _sample_rows(X: np.ndarray, y: np.ndarray, n_target: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) <= n_target:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n_target, replace=False)
    idx = np.sort(idx)
    return X[idx], y[idx]


def _load_concrete(data_dir: Path, n_target: int = 825, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    raw_path = _download_if_needed(UCI_CONCRETE_XLS_URL, data_dir / "raw" / "Concrete_Data.xls")
    try:
        df = pd.read_excel(raw_path)
    except Exception:
        # Pandas may require xlrd for .xls on some setups
        df = pd.read_excel(raw_path, engine="xlrd")
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64)
    return _sample_rows(X, y, n_target, seed)


def _load_energy(data_dir: Path, n_target: int = 615, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    zip_path = _download_if_needed(UCI_ENERGY_ZIP_URL, data_dir / "raw" / "energy+efficiency.zip")
    extract_dir = data_dir / "raw" / "energy_efficiency"
    extract_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = extract_dir / "ENB2012_data.xlsx"
    if not xlsx_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".xlsx"):
                    zf.extract(name, extract_dir)
                    src = extract_dir / name
                    if src != xlsx_path:
                        src.replace(xlsx_path)
                    break
    df = pd.read_excel(xlsx_path)
    # The UCI Energy dataset has 2 targets (Y1=Heating, Y2=Cooling). Paper uses 1 output regression.
    # We choose Y1 (Heating Load) as target by default to keep it single-output.
    X = df.iloc[:, 0:8].to_numpy(dtype=np.float64)
    y = df.iloc[:, 8].to_numpy(dtype=np.float64)
    return _sample_rows(X, y, n_target, seed)


def _load_wine(data_dir: Path, n_target: int = 3919, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    csv_path = _download_if_needed(UCI_WINE_WHITE_URL, data_dir / "raw" / "winequality-white.csv")
    df = pd.read_csv(csv_path, sep=";")
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) > n_target:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=n_target, replace=False)
        df = df.iloc[np.sort(idx)].reset_index(drop=True)
    X = df.drop(columns=["quality"]).to_numpy(dtype=np.float64)
    y = df["quality"].to_numpy(dtype=np.float64)
    return X, y


def _load_abalone(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Abalone dataset (UCI ID 1).
    4177 mẫu, target = Rings (tuổi bào ngư).
    Cột Sex (M/F/I) → one-hot encode → m = 10 features.
    """
    raw_path = _download_if_needed(UCI_ABALONE_URL, data_dir / "raw" / "abalone.data")

    col_names = ["Sex", "Length", "Diameter", "Height",
                 "Whole weight", "Shucked weight",
                 "Viscera weight", "Shell weight", "Rings"]
    df = pd.read_csv(raw_path, header=None, names=col_names)

    # One-hot encode cột Sex (M, F, I) → 3 cột binary
    sex_dummies = pd.get_dummies(df["Sex"], prefix="Sex")
    df = df.drop(columns=["Sex"])
    df = pd.concat([sex_dummies.astype(np.float64), df], axis=1)

    X = df.drop(columns=["Rings"]).to_numpy(dtype=np.float64)  # shape (4177, 10)
    y = df["Rings"].to_numpy(dtype=np.float64)
    return X, y


def _load_auto_mpg(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Auto MPG dataset (UCI ID 9).
    392 mẫu sau khi drop missing, target = mpg.
    m = 7 features: cylinders, displacement, horsepower, weight,
                    acceleration, model_year, origin.
    Bỏ cột car_name (text).
    """
    raw_path = _download_if_needed(UCI_AUTO_MPG_URL, data_dir / "raw" / "auto-mpg.data")

    col_names = ["mpg", "cylinders", "displacement", "horsepower",
                 "weight", "acceleration", "model_year", "origin", "car_name"]
    df = pd.read_csv(raw_path, header=None, names=col_names,
                     sep=r"\s+", na_values="?")

    # Bỏ cột car_name (không phải feature số)
    df = df.drop(columns=["car_name"])

    # Xử lý missing: drop rows có NaN (~6 dòng ở cột horsepower)
    df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=["mpg"]).to_numpy(dtype=np.float64)  # shape (392, 7)
    y = df["mpg"].to_numpy(dtype=np.float64)
    return X, y


PAPER_DATASETS = {
    "Concrete": {"n": 825, "m": 8},
    "Energy": {"n": 615, "m": 8},
    "Wine": {"n": 3919, "m": 11},
}


NOVEL_DATASETS = {
    # Extra exploratory datasets, not part of El Saliby et al. (2024) Scenario 2.
    "Abalone": {"n": 4177, "m": 10},
    "AutoMPG": {"n": 392, "m": 7},
}


ALL_DATASETS = {**PAPER_DATASETS, **NOVEL_DATASETS}


def _cache_matches_expected(cache_path: Path, dataset: str) -> bool:
    if dataset not in ALL_DATASETS or not cache_path.exists():
        return False
    npz = np.load(cache_path)
    expected = ALL_DATASETS[dataset]
    return npz["X"].shape == (expected["n"], expected["m"]) and npz["y"].shape == (expected["n"],)


def prepare_datasets(data_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    cache_dir = data_dir / "preprocessed"
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Dict[str, np.ndarray]] = {}

    for key, loader in [
        ("Concrete", lambda: _load_concrete(data_dir)),
        ("Energy", lambda: _load_energy(data_dir)),
        ("Wine", lambda: _load_wine(data_dir)),
        # Extra exploratory datasets, not part of El Saliby et al. (2024) Scenario 2.
        ("Abalone", lambda: _load_abalone(data_dir)),
        ("AutoMPG", lambda: _load_auto_mpg(data_dir)),
    ]:
        cache_path = cache_dir / f"{key}.npz"
        if _cache_matches_expected(cache_path, key):
            npz = np.load(cache_path)
            X = npz["X"]
            y = npz["y"]
        else:
            X_raw, y_raw = loader()
            X = _standardize_x(X_raw)
            y = _rescale_y_to_minus1_1(y_raw)
            np.savez_compressed(cache_path, X=X, y=y)
        out[key] = {"X": X, "y": y}

    return out


# ============================================================
# ANN regression (tanh, 1 hidden layer)
# ============================================================
@dataclass(frozen=True)
class ProblemSpec:
    dataset: str
    rho_h: int  # 1,2,3
    problem_name: str  # used for reporting


def _theta_dim(m: int, rho_h: int) -> int:
    h = rho_h * m
    return (m * h + h) + (h * 1 + 1)


def _unpack_theta(theta: np.ndarray, m: int, rho_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    h = rho_h * m
    idx = 0
    W1 = theta[idx : idx + m * h].reshape(m, h)
    idx += m * h
    b1 = theta[idx : idx + h]
    idx += h
    W2 = theta[idx : idx + h].reshape(h, 1)
    idx += h
    b2 = float(theta[idx])
    return W1, b1, W2, b2


def _predict(theta: np.ndarray, X: np.ndarray, rho_h: int) -> np.ndarray:
    m = X.shape[1]
    W1, b1, W2, b2 = _unpack_theta(theta, m, rho_h)
    hidden = np.tanh(X @ W1 + b1)
    out = np.tanh(hidden @ W2 + b2)
    return out.reshape(-1)


def make_fitness_fn(X: np.ndarray, y: np.ndarray, rho_h: int):
    def fitness(theta: np.ndarray) -> float:
        pred = _predict(theta, X, rho_h)
        return float(np.mean((pred - y) ** 2))

    return fitness


# ============================================================
# Worker (multiprocessing-safe via global dataset cache)
# ============================================================
_GLOBAL_DATA: Optional[Dict[str, Dict[str, np.ndarray]]] = None


def _init_worker(preprocessed_dir: str):
    global _GLOBAL_DATA
    pre_dir = Path(preprocessed_dir)
    _GLOBAL_DATA = {}
    for key in ALL_DATASETS.keys():
        npz = np.load(pre_dir / f"{key}.npz")
        _GLOBAL_DATA[key] = {"X": npz["X"], "y": npz["y"]}


def _worker(args):
    spec: ProblemSpec
    solver_key: str
    sigma: Optional[float]
    seed: int
    n_evals: int
    spec, solver_key, sigma, seed, n_evals = args

    if _GLOBAL_DATA is None:
        raise RuntimeError("Dataset cache not initialized in worker.")

    X = _GLOBAL_DATA[spec.dataset]["X"]
    y = _GLOBAL_DATA[spec.dataset]["y"]
    dim = _theta_dim(X.shape[1], spec.rho_h)

    fn = make_fitness_fn(X, y, spec.rho_h)

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
            "seed": seed + 1,
            "problem": spec.problem_name,
            "solver_sigma": j_solv,
            "objective": "minimize",
            "best→fitness": r["best_fitness"],
            "genotype_size": dim,
        }
        for r in records
    ]


# ============================================================
# MAIN
# ============================================================
def build_problem_specs() -> List[ProblemSpec]:
    specs: List[ProblemSpec] = []
    for ds in ALL_DATASETS.keys():
        for rho_h in [1, 2, 3]:
            specs.append(ProblemSpec(dataset=ds, rho_h=rho_h, problem_name=f"ea.p.r.{ds}-{rho_h}"))
    return specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_evals", type=int, default=N_EVALS_DEFAULT)
    p.add_argument("--n_rep", type=int, default=N_REP_DEFAULT)
    p.add_argument("--cores", type=int, default=N_CORES_DEFAULT)
    p.add_argument("--quick", action="store_true", help="Run a small subset quickly for smoke test.")
    p.add_argument(
        "--problems",
        nargs="*",
        default=None,
        help="Optional list of problem names to run (e.g. ea.p.r.Concrete-1).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.n_rep = min(args.n_rep, 2)
        args.n_evals = min(args.n_evals, 800)
        args.cores = 1

    # Prepare and cache datasets (preprocessed .npz)
    datasets = prepare_datasets(data_dir)
    preprocessed_dir = (data_dir / "preprocessed").as_posix()

    # Build problems
    specs = build_problem_specs()
    if args.problems:
        wanted = set(args.problems)
        specs = [s for s in specs if s.problem_name in wanted]
        if not specs:
            raise ValueError("No matching problems. Example: ea.p.r.Concrete-1")

    # Tasks
    tasks = []
    for spec in specs:
        for sk in SOLVER_NAME_MAP.keys():
            sig = float(sk.split("-")[1]) if sk.startswith("ES-") or sk.startswith("GA-") else None
            for s in range(args.n_rep):
                tasks.append((spec, sk, sig, s, args.n_evals))

    all_rows = []

    # Use multiprocessing with per-process dataset load to avoid pickling large arrays
    ctx = get_context("spawn")
    if args.cores <= 1:
        _init_worker(preprocessed_dir)
        for t in tqdm(tasks, total=len(tasks)):
            all_rows.extend(_worker(t))
    else:
        with ctx.Pool(processes=args.cores, initializer=_init_worker, initargs=(preprocessed_dir,)) as pool:
            for res in tqdm(pool.imap_unordered(_worker, tasks, chunksize=5), total=len(tasks)):
                all_rows.extend(res)

    out_csv = results_dir / "Scenario_2_Novel_Final.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, sep=";", index=False)

    print(f"Done! Saved to {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
