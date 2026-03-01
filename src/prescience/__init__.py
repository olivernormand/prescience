"""Prescience: AI forecasting evaluation pipeline."""

import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def cmd_fetch(args: list[str]) -> None:
    """Fetch resolved questions from prediction markets and benchmarks."""
    limit = 10
    source = "forecastbench"  # default to forecastbench (best quality questions)
    for i, a in enumerate(args):
        if a in ("--limit", "-n") and i + 1 < len(args):
            limit = int(args[i + 1])
        if a in ("--source", "-s") and i + 1 < len(args):
            source = args[i + 1]

    from .questions import fetch_forecastbench, fetch_polymarket, save_fixtures

    all_questions = []
    if source in ("forecastbench", "all"):
        console.print("Fetching from ForecastBench datasets...")
        all_questions.extend(fetch_forecastbench())
    if source in ("polymarket", "all"):
        console.print("Fetching from Polymarket...")
        all_questions.extend(fetch_polymarket(limit=max(limit * 3, 50)))

    # Select requested number, mixing outcomes
    yes_qs = [q for q in all_questions if q.resolution]
    no_qs = [q for q in all_questions if not q.resolution]

    selected: list = []
    yi, ni = 0, 0
    while len(selected) < limit and (yi < len(yes_qs) or ni < len(no_qs)):
        if yi < len(yes_qs) and (len(selected) % 2 == 0 or ni >= len(no_qs)):
            selected.append(yes_qs[yi])
            yi += 1
        elif ni < len(no_qs):
            selected.append(no_qs[ni])
            ni += 1

    path = save_fixtures(selected)
    console.print(f"[green]Saved {len(selected)} questions to {path}[/green]")

    for q in selected:
        res_str = "[green]Yes[/green]" if q.resolution else "[red]No[/red]"
        console.print(f"  {res_str}  {q.source:>20}  {q.question_text[:70]}")


def cmd_eval(args: list[str]) -> None:
    """Run the full evaluation pipeline."""
    from .eval import evaluate
    from .pipeline import run_pipeline
    from .questions import load_fixtures

    questions = load_fixtures()
    console.print(f"Loaded {len(questions)} questions from fixtures.")

    results = asyncio.run(run_pipeline(questions))
    run_result = evaluate(results)

    # Print results table
    table = Table(title="Evaluation Results")
    table.add_column("Q", style="dim", width=4)
    table.add_column("Question", max_width=50)
    table.add_column("Forecast", justify="right")
    table.add_column("Actual", justify="center")
    table.add_column("Brier", justify="right")

    for i, r in enumerate(run_result.results, 1):
        actual = "[green]Yes[/green]" if r.question.resolution else "[red]No[/red]"
        brier_style = "green" if r.brier_score < 0.1 else "yellow" if r.brier_score < 0.25 else "red"
        table.add_row(
            str(i),
            r.question.question_text[:50],
            f"{r.forecast.probability:.2f}",
            actual,
            f"[{brier_style}]{r.brier_score:.3f}[/{brier_style}]",
        )

    console.print()
    console.print(table)
    console.print(f"\n[bold]Mean Brier Score: {run_result.mean_brier_score:.3f}[/bold]")
    console.print(f"  (0.25 = no skill, 0.10 = good, 0.08 = superforecaster)")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    out_path.write_text(run_result.model_dump_json(indent=2) + "\n")
    console.print(f"\n[dim]Results saved to {out_path}[/dim]")


def cmd_forecast(args: list[str]) -> None:
    """Run a single ad-hoc forecast."""
    if not args:
        console.print("[red]Usage: prescience forecast \"question text\"[/red]")
        sys.exit(1)

    question_text = " ".join(args)

    from .agent import make_agent, parse_forecast

    # Ad-hoc forecasts use today's date (no backtesting restriction)
    agent = make_agent(date.today())
    console.print(f"Forecasting: [bold]{question_text}[/bold]\n")

    result = asyncio.run(agent.run(f"Question: {question_text}\n\nResearch this question and produce your forecast."))
    forecast = parse_forecast(result)

    console.print(f"[bold]Probability: {forecast.probability:.2f}[/bold]")
    if forecast.thinking:
        console.print(f"\n[bold]Thinking:[/bold]\n[dim]{forecast.thinking[:500]}[/dim]")
    console.print(f"\n[bold]Reasoning:[/bold]\n{forecast.reasoning}")


COMMANDS = {
    "fetch": cmd_fetch,
    "eval": cmd_eval,
    "forecast": cmd_forecast,
}


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        console.print("Usage: prescience <command> [args]")
        console.print("Commands:")
        console.print("  fetch     Fetch resolved questions from prediction markets")
        console.print("  eval      Run evaluation pipeline on fixture questions")
        console.print("  forecast  Run a single ad-hoc forecast")
        sys.exit(0)

    cmd = args[0]
    if cmd not in COMMANDS:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        console.print(f"Available: {', '.join(COMMANDS)}")
        sys.exit(1)

    COMMANDS[cmd](args[1:])
