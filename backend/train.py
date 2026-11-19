"""Train the models:  python train.py
Uses real observations from the database when there are at least 500, otherwise synthetic data."""
import pandas as pd

import ml
from db import Observation, engine, init_db

init_db()
real = pd.read_sql_table(Observation.__tablename__, engine)
if len(real) >= 500:
    real["service"] = real["service"].map(lambda s: ml.SERVICES[s]["code"])
    print(f"Training on {len(real)} real observations")
    scores = ml.train(real)
else:
    print("Not enough real data yet, training on synthetic data")
    scores = ml.train()
print("Done:", scores)
