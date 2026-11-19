"""Database setup. Uses SQLite by default; set DATABASE_URL to use PostgreSQL."""
import os
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# Example PostgreSQL URL: postgresql+psycopg2://user:password@localhost:5432/queueless
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///queueless.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


class Token(Base):
    """A person in the virtual queue."""
    __tablename__ = "tokens"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    created: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    served: Mapped[bool] = mapped_column(Boolean, default=False)


class Observation(Base):
    """Real queue readings collected at the pilot site. Used to retrain the models."""
    __tablename__ = "observations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    service: Mapped[str] = mapped_column(String(20))
    hour: Mapped[int] = mapped_column(Integer)
    day_factor: Mapped[float] = mapped_column(Float)
    queue_length: Mapped[int] = mapped_column(Integer)
    counters: Mapped[int] = mapped_column(Integer)
    service_time: Mapped[float] = mapped_column(Float)
    arrivals: Mapped[float] = mapped_column(Float)
    wait_time: Mapped[float] = mapped_column(Float)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
