"""Question -> forecast pipeline runner."""

import asyncio
from datetime import timedelta

from rich.console import Console

from .agent import make_agent, parse_forecast
from .eval import compute_brier_score
from .types import AgentForecast, EvalQuestion, ForecastResult

console = Console()

MAX_CONCURRENCY = 5
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds


async def forecast_question(question: EvalQuestion, semaphore: asyncio.Semaphore) -> ForecastResult:
    """Run the forecasting agent on a single question with concurrency control and backoff."""
    backtest_date = question.backtest_date or (question.resolution_date - timedelta(days=60))

    user_prompt = (
        f"Question: {question.question_text}\n"
        f"Forecast as of: {backtest_date}\n"
        f"Resolution date: {question.resolution_date}\n\n"
        f"Research this question and produce your forecast."
    )

    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                agent = make_agent(backtest_date)
                result = await agent.run(user_prompt)
                forecast = parse_forecast(result)
                brier = compute_brier_score(forecast.probability, question.resolution)
                return ForecastResult(question=question, forecast=forecast, brier_score=brier)
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate" in error_str.lower() or "overloaded" in error_str.lower()
                if is_rate_limit and attempt < MAX_RETRIES - 1:
                    wait = BACKOFF_BASE ** (attempt + 1)
                    console.print(f"  [dim]Rate limited, retrying in {wait:.0f}s...[/dim]")
                    await asyncio.sleep(wait)
                    continue
                raise

    # Unreachable, but satisfies type checker
    raise RuntimeError("Exhausted retries")


async def run_pipeline(questions: list[EvalQuestion]) -> list[ForecastResult]:
    """Run the forecasting pipeline on all questions in parallel with backoff."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def run_one(i: int, question: EvalQuestion) -> ForecastResult:
        console.print(f"\n[bold][{i}/{len(questions)}][/bold] {question.question_text[:80]}")
        try:
            result = await forecast_question(question, semaphore)
            symbol = "[green]✓[/green]" if result.brier_score < 0.25 else "[yellow]~[/yellow]"
            console.print(
                f"  {symbol} p={result.forecast.probability:.2f}  "
                f"actual={'Yes' if question.resolution else 'No'}  "
                f"brier={result.brier_score:.3f}"
            )
            return result
        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]")
            fallback = AgentForecast(
                agent_id="fallback",
                probability=0.5,
                reasoning=f"Agent error: {e}",
            )
            brier = compute_brier_score(0.5, question.resolution)
            return ForecastResult(question=question, forecast=fallback, brier_score=brier)

    tasks = [run_one(i, q) for i, q in enumerate(questions, 1)]
    return list(await asyncio.gather(*tasks))
