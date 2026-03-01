from datetime import date, datetime

from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    question_id: str
    question_text: str
    resolution_date: date
    resolution: bool  # True/False for binary
    source: str  # "metaculus", "polymarket", "forecastbench"
    domain: str | None = None
    backtest_date: date | None = None
    metadata: dict = Field(default_factory=dict)


class AgentForecast(BaseModel):
    agent_id: str
    probability: float = Field(ge=0.0, le=1.0)  # 0 to 1
    reasoning: str  # free-form reasoning trace (visible output)
    thinking: str | None = None  # internal thinking trace (extended thinking)


class ForecastResult(BaseModel):
    question: EvalQuestion
    forecast: AgentForecast
    brier_score: float


class EvalRunResult(BaseModel):
    results: list[ForecastResult]
    mean_brier_score: float
    timestamp: datetime = Field(default_factory=datetime.now)
