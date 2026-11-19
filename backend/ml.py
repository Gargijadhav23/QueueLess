"""QueueLess machine-learning core: data, training, prediction, forecasting, staffing."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).parent / "models"
WAIT_PATH = MODEL_DIR / "wait_model.joblib"
DEMAND_PATH = MODEL_DIR / "demand_model.joblib"

# Peak arrivals per hour and average service minutes for each service point
SERVICES = {
    "college": {"code": 0, "peak": 42, "svc": 6},
    "hospital": {"code": 1, "peak": 30, "svc": 10},
    "bank": {"code": 2, "peak": 55, "svc": 5},
}
# Share of peak demand in each hour of the day
PROFILE = {8: .35, 9: .6, 10: .85, 11: 1, 12: .9, 13: .6, 14: .75, 15: .9, 16: .7, 17: .4, 18: .25}
HOURS = list(PROFILE)
DAY_FACTOR = {0: 1.25, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.1, 5: 0.6, 6: 0.6}  # Monday..Sunday
TARGET_LOAD = 0.8  # keep each counter busy at most 80% of the time

WAIT_FEATURES = ["service", "hour", "day_factor", "queue_length", "counters", "service_time"]
DEMAND_FEATURES = ["service", "hour", "day_factor"]

_models = {}


def make_data(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Synthetic data so the project runs before real queue data exists."""
    rng = np.random.default_rng(seed)
    names = list(SERVICES)
    rows = []
    for _ in range(n):
        s = SERVICES[names[rng.integers(len(names))]]
        hour = int(rng.choice(HOURS))
        day_factor = DAY_FACTOR[int(rng.integers(0, 7))]
        counters = int(rng.integers(1, 9))
        svc = max(1.0, rng.normal(s["svc"], 1.2))
        arrivals = s["peak"] * PROFILE[hour] * day_factor * rng.uniform(0.85, 1.15)
        queue = int(rng.poisson(arrivals * 0.6 * rng.uniform(0.5, 1.5)))
        rho = arrivals / (counters * 60 / svc)
        wait = queue * svc / counters * (1 + 0.25 * max(0.0, rho - 0.7)) + rng.normal(0, 1.5)
        rows.append([s["code"], hour, day_factor, queue, counters, svc, arrivals, max(0.0, wait)])
    return pd.DataFrame(rows, columns=WAIT_FEATURES + ["arrivals", "wait_time"])


def train(df: pd.DataFrame | None = None) -> dict:
    """Train both models, save them, and return accuracy numbers."""
    df = make_data() if df is None else df
    x_tr, x_te, y_tr, y_te = train_test_split(df[WAIT_FEATURES], df["wait_time"], test_size=0.2, random_state=42)
    wait_model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
    wait_model.fit(x_tr, y_tr)
    pred = wait_model.predict(x_te)

    demand_model = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=42, n_jobs=-1)
    demand_model.fit(df[DEMAND_FEATURES], df["arrivals"])

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(wait_model, WAIT_PATH)
    joblib.dump(demand_model, DEMAND_PATH)
    _models.update(wait=wait_model, demand=demand_model)
    return {"mae_minutes": round(mean_absolute_error(y_te, pred), 2), "r2": round(r2_score(y_te, pred), 3)}


def load_models() -> None:
    """Load saved models, or train them the first time."""
    if WAIT_PATH.exists() and DEMAND_PATH.exists():
        _models.update(wait=joblib.load(WAIT_PATH), demand=joblib.load(DEMAND_PATH))
    else:
        train()


def level_of(rho: float) -> str:
    return "Low" if rho < 0.6 else "Moderate" if rho < 0.85 else "High"


def _demand(service: str, day_factor: float) -> np.ndarray:
    x = pd.DataFrame([[SERVICES[service]["code"], h, day_factor] for h in HOURS], columns=DEMAND_FEATURES)
    return _models["demand"].predict(x)


def predict_wait(service, hour, day_factor, queue, counters, svc) -> dict:
    row = pd.DataFrame([[SERVICES[service]["code"], hour, day_factor, queue, counters, svc]], columns=WAIT_FEATURES)
    wait = float(_models["wait"].predict(row)[0])
    arrivals = float(_models["demand"].predict(row[DEMAND_FEATURES])[0])
    rho = arrivals / (counters * 60 / svc)
    return {"wait": round(wait), "lo": round(wait * 0.85), "hi": round(wait * 1.15), "level": level_of(rho)}


def forecast_day(service, day_factor, counters, svc) -> list[dict]:
    out = []
    for h, a in zip(HOURS, _demand(service, day_factor)):
        rho = a / (counters * 60 / svc)
        r = min(rho, 0.95)
        out.append({"hour": h, "arrivals": round(float(a), 1), "load": round(float(rho), 2),
                    "wait": round(float(svc / 2 * r / (1 - r)), 1), "level": level_of(rho)})
    return out


def best_windows(fc: list[dict], k: int = 2) -> list[int]:
    return sorted(x["hour"] for x in sorted(fc, key=lambda x: x["wait"])[:k])


def congestion_alerts(fc: list[dict]) -> list[int]:
    return [x["hour"] for x in fc if x["level"] == "High"]


def staff_plan(service, day_factor, svc, current_counters) -> list[dict]:
    out = []
    for h, a in zip(HOURS, _demand(service, day_factor)):
        need = max(1, int(np.ceil(a * svc / 60 / TARGET_LOAD)))
        out.append({"hour": h, "expected_arrivals": round(float(a), 1),
                    "counters_needed": need, "change": need - current_counters})
    return out
