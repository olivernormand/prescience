import asyncio
from datetime import date

from pydantic_ai import Tool
from valyu import Valyu


def make_valyu_tool(api_key: str, backtest_date: date) -> Tool:
    """Create a Valyu search tool with date filtering.

    Results are restricted to content published on or before backtest_date.
    """
    client = Valyu(api_key=api_key)
    end_date = backtest_date.isoformat()

    async def valyu_search(query: str) -> str:
        """Search the web and proprietary data sources for information relevant to forecasting.

        Use this to find news articles, research papers, and data relevant to the
        forecasting question. Returns titles, URLs, and content snippets.
        Results are filtered to only include content published before the forecast date.
        """
        response = await asyncio.to_thread(
            client.search,
            query=query,
            search_type="all",
            max_num_results=5,
            fast_mode=True,
            end_date=end_date,
        )
        if response is None or not response.success:
            return "Search returned no results."

        parts = []
        for r in response.results:
            content = r.content if isinstance(r.content, str) else str(r.content)
            # Truncate long content
            if len(content) > 2000:
                content = content[:2000] + "..."
            parts.append(f"**{r.title}**\n{r.url}\n{content}\n")

        return "\n---\n".join(parts) if parts else "Search returned no results."

    return Tool(valyu_search)
