import re
from datetime import date
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ThinkingPart

from .config import load_config
from .sources.exa import make_exa_tools
from .sources.valyu import make_valyu_tool
from .types import AgentForecast

OUTPUT_FORMAT_INSTRUCTION = """

## Output Format

After completing your research, write out your full reasoning, then end with your
probability estimate between tags like this:

<probability>0.75</probability>

The probability must be a number between 0 and 1. Do not include anything after
the probability tags.
"""


def _load_superforecasting_prompt() -> str:
    """Load the superforecasting prompt from techniques/ markdown file."""
    path = Path(__file__).resolve().parent.parent.parent / "techniques" / "superforecasting-prompt.md"
    text = path.read_text()

    # Extract content between ``` markers
    parts = text.split("```")
    if len(parts) >= 3:
        return parts[1].strip()

    raise RuntimeError(f"Could not parse superforecasting prompt from {path}")


def make_agent(backtest_date: date) -> Agent[None, str]:
    """Create the forecasting agent with date-filtered search tools.

    Uses str output with extended thinking enabled so we capture both the
    internal thinking trace and the full visible reasoning + probability.
    """
    config = load_config()
    system_prompt = _load_superforecasting_prompt() + OUTPUT_FORMAT_INSTRUCTION

    exa_tools = make_exa_tools(config["EXA_API_KEY"], backtest_date)
    valyu_tool = make_valyu_tool(config["VALYU_API_KEY"], backtest_date)

    return Agent(
        "anthropic:claude-haiku-4-5-20251001",
        output_type=str,
        system_prompt=system_prompt,
        tools=[*exa_tools, valyu_tool],
        model_settings={
            "max_tokens": 16_000,
            "anthropic_thinking": {"type": "enabled", "budget_tokens": 10_000},
        },
        retries=2,
    )


def parse_forecast(result, agent_id: str = "haiku-4.5") -> AgentForecast:
    """Parse the agent's text output into an AgentForecast.

    Extracts the probability from <probability> tags and captures both
    the visible reasoning and the internal thinking trace.
    """
    text: str = result.output

    # Extract probability from tags
    match = re.search(r"<probability>\s*([\d.]+)\s*</probability>", text)
    if match:
        probability = float(match.group(1))
        probability = max(0.0, min(1.0, probability))
        # Reasoning is everything before the probability tag
        reasoning = text[:match.start()].strip()
    else:
        # Fallback: try to find any decimal that looks like a probability
        probability = 0.5
        reasoning = text

    # Extract thinking from message history
    thinking_parts = []
    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ThinkingPart) and part.content:
                thinking_parts.append(part.content)
    thinking = "\n\n".join(thinking_parts) if thinking_parts else None

    return AgentForecast(
        agent_id=agent_id,
        probability=probability,
        reasoning=reasoning,
        thinking=thinking,
    )
