from datetime import date

from exa_py import AsyncExa
from pydantic_ai import Tool


def make_exa_tools(api_key: str, backtest_date: date) -> list[Tool]:
    """Create Exa search and get_contents tools with date filtering.

    Results are restricted to content published on or before backtest_date,
    ensuring no information leakage from after the forecasting cutoff.
    """
    client = AsyncExa(api_key=api_key)
    end_date = backtest_date.isoformat()

    async def exa_search(query: str) -> str:
        """Search the web for information relevant to the forecasting question.

        Returns titles, URLs, published dates, and text content from web pages.
        Results are filtered to only include content published before the forecast date.
        """
        response = await client.search(
            query,
            num_results=5,
            type="auto",
            contents={"text": {"maxCharacters": 3000}},
            end_published_date=end_date,
        )

        parts = []
        for r in response.results:
            text = r.text or ""
            pub = f" ({r.published_date})" if r.published_date else ""
            parts.append(f"**{r.title or 'Untitled'}**{pub}\n{r.url}\n{text}")

        return "\n---\n".join(parts) if parts else "Search returned no results."

    async def exa_get_contents(urls: list[str]) -> str:
        """Get the full text content of specific URLs.

        Use this when you have a URL and want to read its full content.
        """
        response = await client.get_contents(urls, text=True)

        parts = []
        for r in response.results:
            text = r.text or ""
            if len(text) > 3000:
                text = text[:3000] + "..."
            parts.append(f"**{r.title or 'Untitled'}**\n{r.url}\n{text}")

        return "\n---\n".join(parts) if parts else "No content retrieved."

    return [
        Tool(exa_search, name="exa_search"),
        Tool(exa_get_contents, name="exa_get_contents"),
    ]
