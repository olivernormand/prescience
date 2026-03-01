import os
from pathlib import Path

import logfire
from dotenv import load_dotenv

_logfire_configured = False


def configure_logfire() -> None:
    """Configure Logfire observability and instrument pydantic-ai."""
    global _logfire_configured
    if _logfire_configured:
        return
    logfire.configure()
    logfire.instrument_pydantic_ai()
    _logfire_configured = True


def load_config() -> dict[str, str]:
    """Load .env, validate required API keys, and configure observability."""
    # Walk up from this file to find .env at project root
    project_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(project_root / ".env")

    configure_logfire()

    keys = {}
    missing = []
    for name in ("ANTHROPIC_API_KEY", "EXA_API_KEY", "VALYU_API_KEY"):
        val = os.environ.get(name)
        if not val:
            missing.append(name)
        else:
            keys[name] = val

    if missing:
        raise RuntimeError(f"Missing required API keys: {', '.join(missing)}")

    return keys
