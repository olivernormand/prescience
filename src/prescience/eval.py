from datetime import datetime

from .types import EvalRunResult, ForecastResult


def compute_brier_score(probability: float, resolution: bool) -> float:
    """Brier score: (probability - resolution)^2. Lower is better."""
    return (probability - float(resolution)) ** 2


def evaluate(results: list[ForecastResult]) -> EvalRunResult:
    """Compute aggregate evaluation metrics."""
    mean_brier = sum(r.brier_score for r in results) / len(results) if results else 1.0
    return EvalRunResult(
        results=results,
        mean_brier_score=mean_brier,
        timestamp=datetime.now(),
    )
