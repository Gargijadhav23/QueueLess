"""QueueLess API. Run with:  uvicorn main:app --reload"""
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

import ml
from db import Observation, Token, get_db, init_db

app = FastAPI(title="QueueLess API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup() -> None:
    init_db()
    ml.load_models()


class PredictIn(BaseModel):
    service: str = "college"
    hour: int = Field(ge=8, le=18)
    day_factor: float = 1.0
    queue: int = Field(ge=0)
    counters: int = Field(ge=1)
    svc: float = Field(gt=0)


class JoinIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ObservationIn(BaseModel):
    service: str
    hour: int = Field(ge=8, le=18)
    day_factor: float
    queue_length: int = Field(ge=0)
    counters: int = Field(ge=1)
    service_time: float = Field(gt=0)
    arrivals: float = Field(ge=0)
    wait_time: float = Field(ge=0)


def check_service(name: str) -> str:
    if name not in ml.SERVICES:
        raise HTTPException(400, f"Unknown service '{name}'. Use one of {list(ml.SERVICES)}")
    return name


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(d: PredictIn):
    check_service(d.service)
    return ml.predict_wait(d.service, d.hour, d.day_factor, d.queue, d.counters, d.svc)


@app.get("/forecast")
def forecast(service: str = "college", day_factor: float = 1.0, counters: int = 3, svc: float = 6):
    fc = ml.forecast_day(check_service(service), day_factor, counters, svc)
    return {"hours": fc, "best": ml.best_windows(fc), "alerts": ml.congestion_alerts(fc)}


@app.get("/staff")
def staff(service: str = "college", day_factor: float = 1.0, counters: int = 3, svc: float = 6):
    return ml.staff_plan(check_service(service), day_factor, svc, counters)


@app.get("/queue")
def queue(db: Session = Depends(get_db)):
    rows = db.scalars(select(Token).where(Token.served == False).order_by(Token.id)).all()  # noqa: E712
    return [{"no": 100 + t.id, "name": t.name} for t in rows]


@app.post("/queue/join")
def join(d: JoinIn, db: Session = Depends(get_db)):
    token = Token(name=d.name.strip())
    db.add(token)
    db.commit()
    return {"no": 100 + token.id, "name": token.name}


@app.post("/queue/next")
def next_in_queue(db: Session = Depends(get_db)):
    token = db.scalars(select(Token).where(Token.served == False).order_by(Token.id)).first()  # noqa: E712
    if token is None:
        raise HTTPException(404, "The queue is empty")
    token.served = True
    db.commit()
    return {"no": 100 + token.id, "name": token.name}


@app.post("/observations")
def add_observation(d: ObservationIn, db: Session = Depends(get_db)):
    check_service(d.service)
    db.add(Observation(**d.model_dump()))
    db.commit()
    return {"saved": True}
