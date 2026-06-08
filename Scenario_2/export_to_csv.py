import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path("d:/Hoc_Tap/2025-2026/KI_2/Hoc_May/Simple_Evolutionary_Algorithm/Simple_Evolutionary_Algorithm/Scenario_2/data")
pre_dir = data_dir / "preprocessed"
csv_dir = data_dir / "java_csv"
csv_dir.mkdir(exist_ok=True)

datasets = [
    "Concrete",
    "Energy",
    "Wine",
    # Extra exploratory datasets, not part of El Saliby et al. (2024) Scenario 2.
    "Abalone",
    "AutoMPG",
]

for ds in datasets:
    npz_path = pre_dir / f"{ds}.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        X = data["X"]
        y = data["y"]
        # concatenate X and y
        combined = np.column_stack((X, y))
        df = pd.DataFrame(combined)
        out_path = csv_dir / f"{ds}.csv"
        df.to_csv(out_path, index=False, header=False)
        print(f"Saved {out_path}")
    else:
        print(f"Missing {npz_path}")
