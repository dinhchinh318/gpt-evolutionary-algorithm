import pandas as pd

df = pd.read_csv("out/scenario3-progress.csv", sep=";")

rename = {
    "n.iterations": "iterations",
    "n.evals": "evals",
    "n.births": "births",
    "elapsed.secs": "elapsed.seconds",
    "all→each[quality]→uniqueness": "all→each[fitness]→uniqueness",
    "best→quality→behavior.quality→avg.dist": "best→control.quality",
}
df = df.rename(columns=rename)

df["problem"] = df["problem"].str.replace(r"\[avg\.dist\]$", "", regex=True)

solver_map = {
    "CMA-ES": "cmaEs",
    "DE": "de",
    "PSO": "pso",
    "ES-0.02": "simpleEs-0.02",
    "ES-0.25": "simpleEs-0.25",
    "ES-0.5": "simpleEs-0.5",
    "GA-0.02": "ga-0.02",
    "GA-0.25": "ga-0.25",
    "GA-0.5": "ga-0.5",
}
df["solver_sigma"] = df["solver_sigma"].replace(solver_map)

df.to_csv("out/Scenario3.csv", sep=";", index=False)