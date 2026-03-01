"""Fetch resolved binary questions from prediction markets and benchmarks."""

import json
from datetime import date, timedelta
from pathlib import Path

import httpx

from .types import EvalQuestion

FIXTURES_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "questions.json"
POLYMARKET_BASE = "https://gamma-api.polymarket.com"
FORECASTBENCH_BASE = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"

# ForecastBench date sets available (resolution_sets + question_sets)
FORECASTBENCH_DATES = [
    "2025-11-09", "2025-11-23", "2025-12-07", "2025-12-21",
    "2026-01-04", "2026-01-18", "2026-02-01", "2026-02-15",
]

# ForecastBench sources that produce interesting forecasting questions
FORECASTBENCH_INTERESTING_SOURCES = {"manifold", "polymarket", "metaculus", "infer"}


def fetch_polymarket(limit: int = 30, min_volume: float = 50_000_000) -> list[EvalQuestion]:
    """Fetch resolved binary questions from Polymarket's Gamma API."""
    r = httpx.get(
        f"{POLYMARKET_BASE}/markets",
        params={"closed": "true", "limit": limit, "order": "volumeNum", "ascending": "false"},
        timeout=15,
    )
    r.raise_for_status()
    markets = r.json()

    questions: list[EvalQuestion] = []
    for m in markets:
        if m["volumeNum"] < min_volume:
            continue

        outcomes = json.loads(m["outcomes"]) if isinstance(m["outcomes"], str) else m["outcomes"]
        prices = json.loads(m["outcomePrices"]) if isinstance(m["outcomePrices"], str) else m["outcomePrices"]

        # Only binary Yes/No markets
        if len(outcomes) != 2 or "Yes" not in outcomes or "No" not in outcomes:
            continue

        yes_idx = outcomes.index("Yes")
        yes_price = float(prices[yes_idx])

        # Skip ambiguous resolutions (price not clearly 0 or 1)
        if 0.05 < yes_price < 0.95:
            continue

        resolution = yes_price > 0.5

        # Parse resolution date from available fields
        end_iso = m.get("endDateIso") or (m.get("endDate", "")[:10] if m.get("endDate") else None)
        if not end_iso:
            continue
        resolution_date = date.fromisoformat(end_iso)

        # Set backtest_date to ~2 months before resolution
        backtest_date = resolution_date - timedelta(days=60)
        start_iso = m.get("startDateIso")
        if start_iso:
            start_date = date.fromisoformat(start_iso)
            if backtest_date < start_date:
                backtest_date = start_date

        category = ""
        if m.get("events"):
            category = m["events"][0].get("category", "")

        questions.append(
            EvalQuestion(
                question_id=f"pm-{m['id']}",
                source="polymarket",
                question_text=m["question"],
                resolution_date=resolution_date,
                resolution=resolution,
                domain=category or None,
                backtest_date=backtest_date,
                metadata={
                    "polymarket_id": m["id"],
                    "slug": m.get("slug", ""),
                    "volume": m["volumeNum"],
                },
            )
        )

    return questions


def fetch_forecastbench(
    dates: list[str] | None = None,
    sources: set[str] | None = None,
) -> list[EvalQuestion]:
    """Fetch resolved binary questions from ForecastBench datasets.

    ForecastBench provides curated question sets with verified resolutions from
    Metaculus, Polymarket, Manifold, INFER, and data sources (ACLED, FRED, etc).
    """
    dates = dates or FORECASTBENCH_DATES
    sources = sources or FORECASTBENCH_INTERESTING_SOURCES
    seen_ids: set[str] = set()
    questions: list[EvalQuestion] = []

    for date_str in dates:
        try:
            qs_r = httpx.get(f"{FORECASTBENCH_BASE}/question_sets/{date_str}-llm.json", timeout=15)
            qs_r.raise_for_status()
            res_r = httpx.get(f"{FORECASTBENCH_BASE}/resolution_sets/{date_str}_resolution_set.json", timeout=15)
            res_r.raise_for_status()
        except httpx.HTTPError:
            continue

        q_list = qs_r.json()["questions"]
        r_list = res_r.json()["resolutions"]
        q_by_id = {q["id"]: q for q in q_list}

        for r in r_list:
            if not r["resolved"] or r["resolved_to"] not in (0.0, 1.0):
                continue
            if r["source"] not in sources:
                continue
            if r["id"] in seen_ids:
                continue

            q = q_by_id.get(r["id"])
            if not q:
                continue

            seen_ids.add(r["id"])
            resolution_date = date.fromisoformat(r["resolution_date"])

            # Use freeze_datetime as backtest_date (the date the forecast was due)
            freeze = q.get("freeze_datetime", "")
            backtest_date = date.fromisoformat(freeze[:10]) if freeze else resolution_date - timedelta(days=60)

            questions.append(
                EvalQuestion(
                    question_id=f"fb-{r['id'][:12]}",
                    source=f"forecastbench/{r['source']}",
                    question_text=q["question"],
                    resolution_date=resolution_date,
                    resolution=r["resolved_to"] == 1.0,
                    backtest_date=backtest_date,
                    metadata={
                        "forecastbench_id": r["id"],
                        "original_source": r["source"],
                        "question_set_date": date_str,
                        "url": q.get("url", ""),
                    },
                )
            )

    return questions


def load_fixtures() -> list[EvalQuestion]:
    """Load questions from the fixtures file."""
    if not FIXTURES_PATH.exists():
        raise FileNotFoundError(f"No fixtures file at {FIXTURES_PATH}. Run 'prescience fetch' first.")
    raw = json.loads(FIXTURES_PATH.read_text())
    return [EvalQuestion(**q) for q in raw]


def save_fixtures(questions: list[EvalQuestion]) -> Path:
    """Save questions to the fixtures file, merging with any existing ones."""
    FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, EvalQuestion] = {}
    if FIXTURES_PATH.exists():
        for q in json.loads(FIXTURES_PATH.read_text()):
            eq = EvalQuestion(**q)
            existing[eq.question_id] = eq

    for q in questions:
        existing[q.question_id] = q

    merged = sorted(existing.values(), key=lambda q: q.resolution_date, reverse=True)
    FIXTURES_PATH.write_text(
        json.dumps([q.model_dump(mode="json") for q in merged], indent=2) + "\n"
    )
    return FIXTURES_PATH
