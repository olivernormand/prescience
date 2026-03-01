# Technique 3: Search Strategies

**Category:** Inference-time (no model weight changes)
**Prescience stage:** Data acquisition / research agents (§1.4)

## Overview

How agents search for information is the single most important factor in forecast quality. Ablation studies consistently show that removing search collapses accuracy, and that *bad* search actively hurts (worse than no search at all). The difference between a good and bad forecasting system is mostly about search quality.

These are different strategies for how research agents should find and evaluate information. They're implemented as system prompt directives, search tool configurations, and agent personas. A key architectural idea: **agents should have swappable prompt modules** that define their search persona and strategy. We can load in different modules for different agents in the ensemble, giving us the diversity that makes ensembling valuable (see Technique 2c).

---

### 3a. Agentic Adaptive Search (AIA Pattern)

The core pattern from the AIA Forecaster - the first system to match human superforecasters. Rather than generating all search queries upfront, a supervisor agent decides what to search and when, adapting based on what it finds. Each search-reason cycle informs the next query.

**This is already core to Prescience's architecture (§1.4).** The question worth asking: are we diverging from the AIA approach for good reason, or just because? Where AIA has demonstrated something works, we should default to their approach and only diverge when we have evidence for something better. Specifically:

- Adaptive query generation (not all queries upfront) ✓
- Search quality assessment (discard irrelevant results, don't just dump everything in context) ✓
- Multi-agent reconciliation with supervisor ✓
- Statistical extremisation ✓

**Key AIA lesson:** Bad search is worse than no search. Agents need to be selective about what to retrieve and incorporate.

**Beyond standard RAG:** Several advanced retrieval techniques from adjacent ML research are worth watching. MIRAGE-VC (2025) introduces information-gain-driven path retrieval over knowledge graphs, iteratively selecting graph neighbours whose inclusion maximally improves predictor accuracy (+5.0% F1 and +16.6% Precision@5 on venture capital prediction). The core principle - selecting evidence paths by their marginal information gain rather than just relevance - could translate to forecasting by building knowledge graphs of geopolitical actors or economic indicators and using utility-aware path selection. Temporal knowledge graph reasoning (EV-COT, TV-LLM, 2025) combines graph-structural information with LLM reasoning for evolving relationships - potentially useful for questions involving dynamic geopolitical situations.

---

### 3b. Dual-Perspective Retrieval

After forming an initial hypothesis, explicitly search for the strongest evidence against it. Search for both "evidence that X will happen" AND "evidence that X will not happen." This counters the natural tendency for search to confirm the initial impression.

**Application to Prescience:** This is a search directive that goes into the agent's system prompt. Different agents could have different search personas:

- **Confirmation-seeker:** Standard search following the evidence wherever it leads
- **Devil's advocate:** Explicitly searches for disconfirming evidence first
- **Balanced researcher:** Runs parallel searches for and against, then weighs them

This ties into the **swappable prompt module** idea: each agent's search persona is defined by a loadable module that includes its system prompt, search directives, and reasoning style. We can mix and match these across the M agents to get ensemble diversity. The source agreement/disagreement pattern is itself informative - if sources split, that's a signal of genuine uncertainty.

**Evidence clustering (CRAVE, 2025):** Rather than feeding all search results to the agent at once, cluster them into coherent narrative groups (e.g. "bullish evidence" vs. "bearish evidence") and have the agent reason about the competing narratives explicitly. This prevents the agent from being overwhelmed by mixed signals. CONFACT (2025) showed that incorporating source credibility metadata into both retrieval and generation also helps when evidence conflicts.

**Implementation as swappable modules:** Each search persona would be a loadable config that includes system prompt directives (what to search for, how to evaluate results) and tool configurations (which data sources to prioritise). We need to think through how these modules compose with reasoning framework modules (Technique 4) - a search persona + reasoning framework together define the agent's full character.

---

### 3c. Indicators & Warnings

Before searching broadly, generate a structured checklist of specific observable signals that would indicate the event becoming more or less likely. Then systematically check each indicator against data sources.

**Example:** For "Will Country X hold elections by date Y?":
- Leading indicators: constitutional deadline approaching, election commission activity, party registration filings
- Negative indicators: state of emergency declared, opposition boycott announced, constitutional crisis
- For each: where to look, what to expect, what it means for the probability

**Application to Prescience:** The indicator list becomes part of the research plan output from the decomposition step (§1.2). Research agents receive it as a structured checklist alongside the general research plan. This makes search more systematic and reproducible - agents check specific things rather than doing open-ended browsing.

**Particular value for:** Questions with longer time horizons where events unfold over weeks/months. The indicator list provides a systematic way to track developments.

---

## What to Measure

- Do agents with dual-perspective search (3b) produce better-calibrated forecasts than those with standard search?
- Does indicator-based search (3c) improve coverage (finding information that open-ended search misses)?
- Do mixed search personas across the ensemble improve overall Brier score through diversity?

The automated experimentation loop (§2.1.1) should test different combinations of search personas across the agent ensemble.
