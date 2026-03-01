# Prescience - Judgmental Forecasting System Technical Plan

## Overview

This document describes the architecture and implementation plan for an LLM-based judgmental forecasting engine. The system produces calibrated probabilistic forecasts on complex questions by combining agentic search over unstructured data, structured reasoning, backtesting-driven evaluation, and statistical calibration.

The core pipeline has three stages: **data acquisition** (building a state of the world), **reasoning** (generating probability estimates), and **evaluation** (backtesting and calibration). Each stage is designed to be independently testable and improvable, while the overall system is built to be backtestable from day one.

---

## 1. Data Acquisition - Building a State of the World

The system needs to construct, for a given question at a given point in time, a comprehensive information state. This is the single most important component. Ablation studies consistently show that removing search collapses accuracy to worse than always predicting 50%, and that naive search implementations actively hurt. Only agentic, multi-step search helps.

### 1.1 Question Refinement

Before any research begins, the system needs to formalise the input question into a structured format. This is an explicit step, not implicit.

**Question types.** The system should handle two core types:

- **Binary.** "Will X happen by date Y?" Outputs a single probability ∈ [0, 1].
- **Distributional.** "When will X happen?" or "What will Y be worth in 6 months?" Outputs parameters of a probability distribution (e.g. log-normal, mixture) over the outcome space. Binary questions are a special case: they're just a CDF evaluated at a specific point.

The refinement step takes a natural-language question and produces:

- Explicit question type classification
- Formalised resolution criteria (what counts as "yes"?)
- Resolution date or time horizon
- Key ambiguities flagged for the user or for the system to resolve

**Implementation.** A single LLM call with structured output. The model should have access to a lightweight search tool at this stage - it may need to look up entities, check whether a question is already being tracked on prediction markets, or clarify ambiguous references. This isn't the deep research phase, but a quick "does this question make sense and what exactly are we forecasting?" check.

```python
class FormalisedQuestion(BaseModel):
    original_question: str
    refined_question: str
    question_type: Literal["binary", "distributional"]
    resolution_criteria: str
    resolution_date: date
    background: str  # brief context from lightweight search
```

### 1.2 Search Architecture

#### Search API Strategy

We want two tiers of search capability: backtestable search (APIs with strict date filtering, essential for evaluation) and best-available search (highest quality results for live forecasting, regardless of backtestability).

**Starting implementations:**

- **Valyu** (`pip install valyu`) — Primary search. Unified access to web + proprietary sources (SEC filings, academic papers, financial data). Date filtering via `start_date`/`end_date` parameters (YYYY-MM-DD format). Backtestable. Python SDK with `x-api-key` header auth.
- **Exa** (`pip install exa-py`) — Secondary search. Own search index, sub-500ms latency, strong semantic search. Research endpoint does multi-step agentic search with structured output. Date filtering via `startPublishedDate`/`endPublishedDate` (ISO 8601). Partially backtestable (dates are estimated publication dates). `AsyncExa` client for async support.

**Future additions:**

- **Tavily** — Popular, solid quality, good agentic integration. No backtestable version.
- **NewsCatcher** — 120,000+ global news sources, NLP enrichment (sentiment, entities), near-real-time. CatchAll product is a recall-first web search API. Good for dedicated news coverage.
- **NewsAPI.ai** — 150,000+ sources, 60+ languages, archive back to 2014. Strong Boolean filtering and concept-based search. Potentially good for backtesting given historical depth.
- **Perigon** — AI-powered news API with event-level queries and advanced filtering.

**Critical concern for backtesting:** We need to understand which APIs are vulnerable to foresight bias - i.e., they leak new information into old search results. This can happen when articles are updated post-publication, when search indices are rebuilt with newer content, or when result ranking algorithms use signals from future engagement. This should be a dedicated investigation before we trust any API for backtesting.

#### The Pluggable Data Source Pattern

All external data sources - search APIs, news APIs, prediction markets, structured data feeds, web scrapers - should conform to a common interface so they can be easily swapped in and out. The pattern:

```python
class DataSource(ABC):
    """Base class for all external data sources available to research agents."""

    @abstractmethod
    async def query(
        self,
        query: str,
        backtest_date: datetime | None = None,
        **kwargs,
    ) -> list[SourceResult]:
        """
        Execute a query against this data source.

        If backtest_date is set, only return results that were available
        before that date. If the source doesn't support date filtering,
        raise NotBacktestable.

        Each implementation accepts source-specific kwargs (e.g. num_results,
        topic filters, language, region). These are documented per-source
        so the research agents know what parameters are available and how
        to use them effectively.
        """
        ...

    @property
    @abstractmethod
    def supports_backtest(self) -> bool:
        """Whether this source supports date-filtered queries."""
        ...

    @property
    @abstractmethod
    def source_type(self) -> str:
        """e.g. 'web_search', 'news', 'prediction_market', 'structured_data', 'web_scraper'"""
        ...

    @property
    @abstractmethod
    def usage_guide(self) -> str:
        """
        Human-readable documentation for how to use this source effectively.
        Included in the research agent's system prompt so it knows what
        parameters to pass, what queries work well, and what to expect.
        We write this rather than letting the model discover it by trial and error.
        """
        ...
```

**Starting implementations:**

- `ValyuSearch(DataSource)` - web + proprietary, backtestable via date params
- `ExaSearch(DataSource)` - web search, partially backtestable

**Future additions:**

- `TavilySearch(DataSource)` - web search, not backtestable
- `NewsCatcherNews(DataSource)` - news, backtestable with date filtering
- `PolymarketFeed(DataSource)` - prediction market prices, backtestable (historical price data)
- `MetaculusFeed(DataSource)` - prediction market, backtestable
- `FREDData(DataSource)` - economic time series, backtestable
- `ACLEDData(DataSource)` - conflict events, backtestable
- `PlaywrightScraper(DataSource)` - general web scraping via Playwright, not backtestable
- `PrivateDataSource(DataSource)` - hook for user-provided private/internal data (see §5)

Each source returns `SourceResult` objects. The `backtest_date` parameter is the key mechanism for the backtesting pipeline: when set, all sources must either return only pre-date results or raise an exception indicating they can't comply.

```python
class SourceResult(BaseModel):
    content: str
    url: str | None = None
    published_date: datetime | None = None
    retrieved_date: datetime
    source_name: str
```

Each source also provides a `usage_guide` - structured documentation of its capabilities, supported parameters, and best practices for querying it. These guides are injected into the research agent's context so the model knows how to use each source well, rather than having to discover effective query patterns through trial and error.

There's no limit to how many sources we include. Adding a new data feed should be as simple as extending the base class and registering it in the agent's tool registry.

### 1.3 Research Agents

Each research run spawns M independent research agents (start with M=5, scale to M=10). Each agent receives the `FormalisedQuestion` and has access to all registered data sources as tools. Agents plan their own research autonomously - there is no separate decomposition step.

**Research planning through prompting.** The agent's system prompt encodes superforecasting methodology as guidance: identify relevant sub-questions, look up reference classes and base rates, track key actors and causal factors, consider conditional dependencies, check specific data sources (prediction markets, structured feeds). These are hints, not structured output requirements - the agent is free to plan its research however it judges most productive. There will always be edge cases and novel question types where rigid decomposition fails, so we leave this to the agent's judgement.

**Coupled search + reasoning.** Each agent both searches and reasons within its loop. The agent searches, reads, forms a tentative view, identifies gaps, searches for those gaps, and repeats. It then produces a probability estimate alongside its reasoning. This mirrors how skilled human researchers actually work - hypothesis formation guides further research.

**Future addition: Separated search + reasoning.** An alternative where one set of agents does deep research only (producing information briefings without predictions), and a separate set of reasoning agents reads these briefings and produces forecasts without search access. Cleaner separation of concerns, but research agents don't know what reasoners will need. Will be added as a configurable option once the coupled baseline is working.

**Light touch on the loop itself.** We should not over-engineer the agentic search loop. Frontier models are increasingly capable at multi-step tool use, and the marginal value of elaborate loop orchestration is declining. Each agent should:

1. Receive the formalised question and available tools
2. Execute its research autonomously, following whatever path it judges most productive
3. Report back with its probability estimate and free-form reasoning trace

**Sub-agent delegation.** For complex questions, agents can spawn sub-agents via a `delegate_to_subagent(goal: str)` tool call. This is how question decomposition happens in practice - an agent decides a question needs to be broken down, spawns sub-agents with specific research goals, and synthesises their findings. This is supported but not required; simple questions won't need it.

Before building custom deep-research orchestration, we should evaluate off-the-shelf solutions - there's significant existing work in agentic research (e.g. deep research capabilities from model providers, Exa's research endpoint, open-source research agent frameworks). If an off-the-shelf tool produces high-quality research briefings, we should use it as a data source or agent backbone rather than duplicating that work. Where we do build our own, we should reuse existing agentic patterns (e.g. from the Claude Agents SDK or Pydantic AI's agent orchestration).

**Future addition: Structured decomposition step.** If we find that agents' autonomous research planning is insufficient - e.g. they consistently miss important sub-questions or fail to check obvious reference classes - we can reintroduce a structured decomposition step as a pipeline stage between refinement and research. This would produce a typed research plan that agents follow. For now, we start without it.

**Parallel execution.** All M research agents run in parallel. Within each agent, tool calls should also be parallelised where possible (e.g. querying multiple data sources simultaneously).

### 1.4 Observability

Full observability over the research pipeline is essential, both for debugging and for understanding what drives forecast quality.

**Framework.** Pydantic Logfire (OpenTelemetry-based). Every agent run, tool call, and LLM interaction should be traced.

**Trace structure.** Traces should be linked to:

- **Question ID:** A deterministic hash of the formalised question (e.g. `q_a3f8c2d1`). This is the primary key for filtering - you should be able to see all runs and sub-runs for a given question.
- **Run ID:** A unique identifier for each full forecasting run (e.g. `run_20260228_001`). One question can have multiple runs (e.g. when re-forecasting with updated information).
- **Agent ID:** Which of the M research agents this trace belongs to (e.g. `q_a3f8c2d1/run_001/agent_03`).
- **Sub-agent ID:** If an agent spawns sub-agents, those get nested IDs (e.g. `q_a3f8c2d1/run_001/agent_03/sub_01`).

This hierarchy lets you filter at any level: see all traces for a question, drill into a specific run, inspect what a specific agent searched for and found, or trace a sub-agent's execution.

**What to log:** Query text, search results (with source URLs and dates), LLM prompts and completions, tool call inputs/outputs, timing, token counts, costs.

---

## 2. Reasoning - Generating Probability Estimates

### 2.1 Individual Agent Forecasts

Each of the M research agents, having completed its research, produces an `AgentForecast`:

```python
class AgentForecast(BaseModel):
    agent_id: str
    probability: float  # 0 to 1
    reasoning: str  # free-form reasoning trace
```

The reasoning trace is unconstrained free-form text. The agent's system prompt hints at good forecasting practices - start with base rates, consider inside and outside views, list factors for and against, avoid anchoring on recency - but the model is free to structure its reasoning however it wants. We don't force the LLM into a rigid schema that might reduce reasoning quality.

**Future addition:** Structured fields (factors_for, factors_against, base_rate, base_rate_source, evidence items, confidence/meta-uncertainty) as optional fields on `AgentForecast`, if downstream aggregation or error diagnostics (Technique 6) benefit from structured decomposition of reasoning.

Because each agent couples search and reasoning (§1.3), these forecasts are naturally diverse - different agents found different information and formed different views.

**Prompting strategy.** The prompt should encode forecasting best practices, but the entire prompting strategy is an experimental surface we expect to iterate on heavily. These are hypotheses, not fixed rules. Different prompting strategies may work better for different question types or domains. The system should make it easy to swap prompt templates and measure the impact on calibration.

**Reasoning trace updates.** When new information arrives after an initial forecast (e.g. breaking news, a new data release, a market move), the system should not re-run the entire pipeline from scratch. Instead, the new evidence should be used to *update the existing reasoning trace*. The agent receives its prior reasoning trace and the new evidence, and produces an updated trace with a revised probability. This is cheaper, faster, and preserves the reasoning history - you can see how each piece of evidence shifted the estimate. It also connects directly to path independence (Technique 10a): a well-calibrated update process should produce the same final probability regardless of whether the evidence arrived incrementally or all at once. If it doesn't, that's a diagnostic signal. This is a future capability - the initial system produces one-shot forecasts - but the architecture should not preclude it. The `AgentForecast` reasoning trace should be designed so that it can be fed back into an agent as prior context for an update round.

**Known LLM failure modes to mitigate:**

- Universal overconfidence: LLMs produce overconfident predictions across domains
- Extended reasoning paradox: longer reasoning chains can worsen calibration by generating persuasive arguments that inflate confidence
- Anchoring on recency: excessive weight on recent news over base rates
- Hedging toward 50%: systematic pull toward maximum uncertainty (corrected in calibration, §2.3)
- Domain-specific weaknesses: consistently worse on economics/finance than political forecasting
- Conjunction fallacy: underperform on conjunction questions and unprecedented events

### 2.1.1 Automated Experimentation Loop

A key design goal: it should be trivial to run the full system in backtested mode and get a Brier score out. This enables a closed eval loop where an autonomous agent (e.g. Claude Code) can:

1. Formulate a hypothesis about what might improve performance (e.g. a different prompt strategy, a different number of agents, a different aggregation method)
2. Modify the relevant configuration or prompt
3. Run a backtested evaluation against a standard question set
4. Compare the result to the baseline
5. Accept or reject the change based on the score delta
6. Repeat

This is one of the most important capabilities of the system. If the eval loop is easy to run, we can parallelise experimentation across many hypotheses simultaneously. The system should be designed so that everything that matters for forecast quality - prompts, agent count, aggregation method, calibration parameters, data source configuration - is configurable and measurable through this loop. Every experiment should be logged with full provenance so we can trace what changed and why.

### 2.2 Aggregation

The aggregation step takes M individual agent forecasts and produces one or more reconciled probability estimates. This is implemented as a pluggable interface - different aggregators can run in parallel within the same graph, producing multiple forecast outputs that can be compared.

```python
class Aggregator(ABC):
    """Base class for forecast aggregation strategies."""

    @abstractmethod
    async def aggregate(
        self, forecasts: list[AgentForecast], ctx: GraphRunContext,
    ) -> AggregatedForecast:
        """Take M individual forecasts and produce an aggregated estimate."""
        ...
```

Each aggregator produces an `AggregatedForecast`:

```python
class AggregatedForecast(BaseModel):
    aggregator_id: str
    probability: float
    reasoning: str  # free-form — how the aggregation was decided
    agent_forecasts: list[AgentForecast]  # the inputs
```

**Starting implementations:**

- **MeanAggregator** — Simple mean of M probabilities. The baseline. Surprisingly strong because the diversity of search paths across agents does most of the work.
- **SupervisorAggregator** — An LLM-based aggregator that receives all M forecasts with their reasoning traces and produces a reconciled estimate. It can identify areas of agreement/disagreement, check for systematic biases, and do targeted additional search to resolve factual disputes. Should have access to explicit bias-checking prompts, e.g.:
  - "The ensemble median is X. List three reasons this could be too high. List three reasons this could be too low."
  - "Are any of the individual forecasts anchored on a single recent news article?"
  - "What base rate would an uninformed forecaster use? How far is the ensemble from that base rate, and is the deviation justified?"

**Future additions:**

- **TrimmedMeanAggregator** — Remove outlier agents before averaging.
- **ExtremisedMeanAggregator** — Mean pushed toward extremes (see calibration, §2.3).

The system should support running multiple aggregators on the same set of agent forecasts in a single pipeline run. This lets us compare aggregation strategies head-to-head on every question, not just in aggregate across an eval suite. Each aggregator produces its own forecast output, and evaluation can track which aggregator performs best over time.

### 2.3 Statistical Calibration

An optional final transformation applied to the aggregated probability. This step can be toggled on or off per run. LLMs systematically hedge toward 50%, so the calibration step pushes probabilities toward the extremes where evidence supports it.

**Platt scaling.** Fit a logistic regression on historical forecast-outcome pairs:

$$P(\text{true}) = \frac{1}{1 + e^{A \cdot \text{logit}(p) + B}}$$

where p is the system's raw probability. Parameters A and B are learned from a calibration dataset of past forecasts and their resolutions. A common approach is extremisation with α=√3.

**Implementation.** Two lines of code once you have calibration data. The hard part is accumulating enough resolved forecasts to fit the curve reliably (~100-200 across the probability range).

**Bootstrap approach:** Use historical benchmark datasets (e.g. resolved questions from Metaculus, Polymarket, or other public question sets) to generate the system's forecasts retroactively and fit initial calibration before we have live data.

**Toggleable by design.** Calibration should be a clean on/off switch in the pipeline configuration. This lets us measure whether it helps or hurts for a given model + aggregator combination, and makes it easy to disable as models improve. Worth tracking whether the calibration step continues to add value over time, or whether it converges toward an identity transform as frontier models become better calibrated out of the box.

---

## 3. Evaluation - Backtesting and Metrics

### 3.1 Why Backtest

Backtesting serves three distinct purposes:

1. **Performance prediction.** Know in advance roughly how well the system will perform on live forecasting, before waiting months for events to resolve.
2. **RL training data.** Generate training signal for reinforcement learning on open-weight models (§4).
3. **Harness fitting.** Tune the system's prompts, calibration, and data source configuration on historical data using one model generation, then reapply the optimised harness to new model updates. When a new frontier model drops, you can immediately deploy it into the validated harness.

### 3.2 The Knowledge Cutoff Approach

Use a model with a known knowledge cutoff (e.g. an open-weight model trained on data through July 2024) to forecast events that have since resolved. The model genuinely doesn't know what happened after its cutoff, so you're testing real forecasting ability rather than recall.

**Pitfalls to mitigate:**

*Temporal leakage.* Future information can leak through:
- The questions themselves: questions generated from 2025 news about late 2024 events carry specificity that reveals the outcome
- Search APIs returning articles updated after the cutoff date
- Models trained on data past their nominal cutoff

*Survivorship bias.* Questions generated from current data systematically miss entities that no longer exist, creating a skewed distribution toward "things that survived."

*Logical leakage.* If a model infers it's being backtested as of date T, it can sometimes logically deduce the answer (e.g. "if you can grade my prediction about whether X happened before 2025, then we've reached 2025").

### 3.3 Foreknowledge Detection

**LLM-as-judge filter.** A separate system - distinct from the forecasting agents - that reviews search results for evidence of foreknowledge relative to the backtest date. This runs as a dedicated filtering layer: every search result returned during a backtesting run passes through the judge, which flags or removes results that contain information from after the backtest date.

**Implementation.** The `backtest_date: datetime | None` parameter on the `DataSource` base class (§1.2) is the primary mechanism. When set:

- Backtestable sources apply strict date filtering on original publication date
- Non-backtestable sources raise `NotBacktestable` and are excluded from the run
- All results that pass through are additionally screened by the LLM-as-judge for subtle foreknowledge (e.g. an article published before the cutoff that was edited after it)

**API foresight audit.** We should conduct a systematic evaluation of which search APIs are vulnerable to foresight bias. For each candidate API, run a set of historical queries with date filters and check whether any results contain information that wasn't available at the specified date. Document the results so we know which APIs we can trust for backtesting.

### 3.4 Evaluation Dataset - Question Generation

This is one of the hardest problems in the space. The quality of the evaluation dataset directly determines whether backtesting results are meaningful.

```python
class EvalQuestion(BaseModel):
    question_id: str
    question_text: str
    resolution_date: date
    resolution: bool | float  # True/False for binary, value for distributional
    source: str  # "metaculus", "polymarket", "forecastbench"
    domain: str | None = None
    metadata: dict = {}  # source-specific fields (community prediction, market price, etc.)
```

**Primary sources for evaluation questions:**

1. **Metaculus historical questions** (starting point). REST API at `api.metaculus.com/api/posts/`, with `?statuses=resolved&forecast_type=binary` for resolved binary questions. Token auth via `Authorization` header. ~4,000+ resolved questions with clear resolution criteria, verified outcomes, and community predictions available for comparison. Best available source for clean evaluation data.
2. **ForecastBench.** ICLR 2025 benchmark with 5,948 questions. Available via HuggingFace datasets. Published superforecaster baselines (Brier ~0.093) and biweekly question releases for ongoing evaluation.
3. **Polymarket resolved markets.** Gamma API for resolved markets with historical price data. Different distribution of question types (more crypto/politics, fewer science/tech).

**Sampling considerations:** These sources have different distributions over question types, difficulty levels, and domains. Naively combining them produces a biased dataset. We should:

- Categorise all questions by domain (geopolitics, economics, science, technology, culture, etc.)
- Sample to ensure reasonable coverage across domains
- Track and report performance by domain, not just aggregate
- Be transparent about the composition of the eval set

**Auto-generated questions.** Supplement with questions generated from structured data sources (economic time series, conflict event databases, Wikipedia) using template approaches. These are useful for coverage but need human validation to confirm they're reasonable questions.

**Question generation for users.** Beyond evaluation, question generation is also a product concern. Users often don't know what forecasts they need - they have a topic or a decision and need help formulating the right questions. This is a separate pipeline that takes a topic or decision context and generates a set of relevant forecasting questions: given a decision, what questions would a superforecaster want answered before making that decision? This is distinct from the eval dataset generation but shares underlying technology.

### 3.5 Evaluation Tooling

The evaluation pipeline should be simple to run and produce consistent, comparable outputs.

**Core interface:**

```python
class EvalRun:
    """A single evaluation run across a set of questions."""
    
    def __init__(self, questions: list[EvalQuestion], forecasts: list[Forecast]):
        ...
    
    def brier_score(self) -> float:
        """Aggregate Brier score: (1/N) Σ(f_i - o_i)²"""
        ...
    
    def calibration_curve(self) -> CalibrationData:
        """Predicted probability vs actual frequency, binned."""
        ...
    
    def domain_breakdown(self) -> dict[str, DomainMetrics]:
        """Brier score, calibration, resolution broken down by question domain."""
        ...
    
    def save(self, path: str):
        """
        Save full evaluation output:
        - brier_score.json (aggregate + per-question)
        - calibration_plot.png 
        - domain_breakdown.json
        - full_results.jsonl (every question, forecast, outcome, reasoning)
        """
        ...
```

**Metrics to compute:**

- **Brier score:** (1/N)Σ(f_i - o_i)². Ranges 0 (perfect) to 1 (worst). Baseline (always 50%) = 0.25. Top human superforecasters ≈ 0.08, best LLM systems ≈ 0.10-0.11.
- **Calibration:** Predicted probability vs actual frequency. Visualise with a calibration plot. Report Expected Calibration Error (ECE).
- **Resolution:** How much predictions vary - are we discriminating between likely and unlikely events, or just predicting 50% on everything?
- **Domain breakdown:** Performance by question category (geopolitical, economic, science, etc.). LLMs are notably weaker on economics/finance.

Every eval run should be saved with full provenance: which model, which search APIs, which agent configuration, which calibration parameters. This makes it possible to compare runs and track improvement over time.

---

## 4. Reinforcement Learning and Fine-Tuning

### 4.1 Strategic Position

The default bet should be to use the best available frontier models with good scaffolding. The scaffolding (search architecture, prompt engineering, calibration, tool access) is where most of the performance comes from. When a new frontier model is released, it can be immediately deployed into the existing harness.

RL fine-tuning makes sense under specific conditions:

- **If retrieval can be part of the RL loop.** The most valuable thing to learn through RL is not just "how to reason about forecasting" but "how to search effectively for forecasting." If the model can make search tool calls during RL rollouts and receive reward signal based on the quality of its final forecast, it learns to search better - which information sources to check, what follow-up queries to run, when to stop searching. This is the only way to incentivise correct search behaviour through training. Without retrieval in the loop, RL only improves reasoning on already-retrieved information, which is less valuable.
- **If you can train on top of frontier models without delay.** The fundamental challenge: to generate RL training data, you need resolved questions, which means you need a gap between when the questions were asked and when you train. During that gap, a new frontier model may have been released that matches or exceeds your fine-tuned model out of the box. Unless you can find a paradigm for generating training signal that doesn't require this delay (e.g. consistency-based rewards, synthetic calibration targets), RL will always be chasing the frontier.

### 4.2 Implementation Approach

If we do pursue RL, we should use existing tooling rather than building from scratch:

- **OpenPipe ART** (Agent Reinforcement Trainer) - purpose-built for multi-turn agentic RL using GRPO. Separates the training backend from the agent frontend, so you can embed RL into existing codebases. Supports multi-step tool use in rollouts, which is exactly what we need for retrieval-augmented forecasting.
- **Weights & Biases** - for experiment tracking and comparison across RL runs.

The backtesting infrastructure (§3) generates the reward signal: Brier score on backtested questions. The RL environment is the forecasting pipeline itself - model receives a question, makes search tool calls, produces a forecast, receives Brier score as reward.

**Open-weight model selection:** For RL, we need an open-weight model. The choice should be whatever is closest to frontier at the time we're ready to train. The harness and evaluation infrastructure should be model-agnostic so we can swap models easily.

---

## 5. Private Data Integration

The forecasting engine described above operates on public information. For many use cases, the real value comes from combining external forecasting with internal organisational data - proprietary market data, internal memos, CRM data, supply chain information.

This should be implemented as an additional `DataSource` (§1.2):

```python
class PrivateDataSource(DataSource):
    """
    Hook for user-provided private/internal data.
    
    Implementations might include:
    - A vector store over internal documents
    - An API to query internal databases
    - A file system of uploaded documents
    - Integration with tools like Notion, Google Drive, Confluence
    """
    
    supports_backtest = False  # It's likely that private data is inherently not backtestable.
    source_type = "private"
```

The research agents treat private data sources exactly like any other tool - they can query them alongside public search, prediction markets, and structured feeds. The system prompt should encourage agents to check internal sources when the question is relevant to the user's domain.

---

## 6. Architecture

### 6.1 System Overview

The diagram below shows the full pipeline. Boxes marked with `[*]` indicate points of optionality - configurable steps that can be swapped, toggled, or run in parallel with alternatives.

```
┌─────────────────────────────────────────────────────────────┐
│                     QUESTION INPUT                          │
│                   (natural language)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  QUESTION REFINEMENT                        │
│        Classify type · Formalise resolution criteria         │
│              Lightweight search for grounding                │
│                → FormalisedQuestion                          │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┼──────────┐
              │          │          │          ... M agents
              ▼          ▼          ▼
┌───────────────┐┌───────────────┐┌───────────────┐
│  RESEARCH     ││  RESEARCH     ││  RESEARCH     │
│  AGENT 1      ││  AGENT 2      ││  AGENT M      │
│               ││               ││               │
│  Autonomous   ││  Autonomous   ││  Autonomous   │
│  research     ││  research     ││  research     │
│  planning     ││  planning     ││  planning     │
│               ││               ││               │
│  Search ↔     ││  Search ↔     ││  Search ↔     │
│  Reason       ││  Reason       ││  Reason       │
│     ↓         ││     ↓         ││     ↓         │
│ AgentForecast ││ AgentForecast ││ AgentForecast │
└───────┬───────┘└───────┬───────┘└───────┬───────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │     [*] Multiple aggregators
              ▼                     ▼         run in parallel
     ┌──────────────┐    ┌───────────────┐
     │     Mean     │    │  Supervisor   │
     │  Aggregator  │    │  Aggregator   │
     └──────┬───────┘    └───────┬───────┘
            │                    │
            ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│             [*] CALIBRATION (optional)                       │
│          Platt scaling · Extremisation · Identity            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  FORECAST OUTPUT(S)                          │
│       One per aggregator · ForecastOutput                    │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Framework Choice

**Pydantic AI** with **pydantic-graph** for the overall agent orchestration. Rationale:

- Type-safe structured outputs (essential for forecast formats)
- Built-in graph support for defining the multi-step pipeline as a state machine
- `GraphRunContext` as shared state - all nodes in the pipeline operate on a common typed context object
- Native Logfire integration for observability (§1.4)
- Model-agnostic: supports all major providers (focusing on OpenAI and Anthropic), making it easy to swap frontier models
- MCP support for tool integration
- Durable execution for long-running research agents
- Active development by the Pydantic team (Sequoia-backed)

The pipeline maps naturally onto a pydantic-graph:

```
[Refinement] → [ParallelResearch] → [Aggregation] → [Calibration?] → [End]
```

Each node is a `BaseNode` subclass with typed state via `GraphRunContext`. The `ParallelResearch` node spawns M concurrent agent runs. The `Aggregation` node runs all configured aggregators, each producing its own forecast. The `Calibration` node is optional and toggled by configuration. The graph can be interrupted and resumed (useful for long research runs or human-in-the-loop review).

**Pipeline state and dependencies:**

```python
@dataclass
class ForecastState:
    """Mutable state that flows through the pipeline graph."""
    question: FormalisedQuestion | None = None
    agent_forecasts: list[AgentForecast] = field(default_factory=list)
    aggregated_forecasts: list[AggregatedForecast] = field(default_factory=list)
    final_output: ForecastOutput | None = None

@dataclass
class PipelineDeps:
    """Immutable dependencies injected into all graph nodes."""
    model: Model
    search_sources: list[DataSource]
    aggregators: list[Aggregator]
    config: PipelineConfig
```

**Final output:**

```python
class ForecastOutput(BaseModel):
    question: FormalisedQuestion
    forecasts: list[AggregatedForecast]  # one per aggregator
    metadata: dict  # run_id, timestamps, model info, costs
```

### 6.3 Terminal Interface

For now, the primary interface is a terminal CLI. Configuration is controlled via CLI flags that map to the config schema (§6.5).

```bash
# Single forecast
prescience forecast "Will X happen by Y?" --agents 5 --model openai:gpt-4o

# Backtested evaluation
prescience eval --dataset metaculus --backtest-date 2024-07-01 --agents 5

# Compare aggregation strategies
prescience eval --compare --aggregator mean,supervisor

# Inspect a past run
prescience inspect <run_id>
```

The CLI should produce clean, readable output - formatted probabilities, reasoning summaries, and links to full trace data in Logfire. This is the interface we'll use day-to-day for development and experimentation, so it needs to be pleasant to work with.

### 6.4 Project Structure

```
prescience/
├── src/
│   ├── questions/
│   │   ├── refine.py          # Question refinement (§1.1)
│   │   └── types.py           # FormalisedQuestion, AgentForecast, AggregatedForecast, etc.
│   ├── research/
│   │   ├── agent.py           # Individual research agent (§1.3)
│   │   └── prompts/           # System prompts for research + forecasting
│   ├── sources/
│   │   ├── base.py            # DataSource ABC, SourceResult (§1.2)
│   │   ├── search/
│   │   │   ├── valyu.py       # ValyuSearch (primary)
│   │   │   └── exa.py         # ExaSearch (secondary)
│   │   ├── markets/           # Polymarket, Metaculus feeds (future)
│   │   ├── structured/        # FRED, ACLED, Yahoo Finance (future)
│   │   └── private.py         # Private data source hook (§5)
│   ├── aggregation/
│   │   ├── base.py            # Aggregator ABC (§2.2)
│   │   ├── mean.py            # MeanAggregator
│   │   └── supervisor.py      # SupervisorAggregator (LLM-based)
│   ├── calibration/
│   │   └── calibrate.py       # Platt scaling, extremisation (§2.3)
│   ├── pipeline/
│   │   ├── graph.py           # Pydantic-graph pipeline definition
│   │   ├── state.py           # ForecastState, PipelineDeps
│   │   ├── runner.py          # Single question forecast runner
│   │   └── batch.py           # Batch forecasting over multiple questions
│   ├── eval/
│   │   ├── metrics.py         # Brier, calibration, resolution, domain breakdown
│   │   ├── run.py             # EvalRun class (§3.5)
│   │   ├── questions/         # Eval question loaders (Metaculus, Polymarket, ForecastBench)
│   │   └── plots.py           # Calibration plots, domain breakdowns
│   ├── backtest/
│   │   ├── engine.py          # Backtesting orchestrator
│   │   ├── foreknowledge.py   # LLM-as-judge foreknowledge filter (§3.3)
│   │   └── foresight_audit.py # API foresight bias evaluation
│   ├── cli.py                 # Terminal interface (§6.3)
│   └── config.py              # PipelineConfig and sub-configs (§6.5)
├── evals/
│   ├── datasets/              # Evaluation question sets
│   ├── results/               # Saved eval runs
│   └── scripts/               # Scripts to generate/sample eval datasets
├── tests/
└── pyproject.toml
```

### 6.5 Configuration

All pipeline behaviour is controlled through a typed config that maps directly to CLI flags.

```python
class ModelConfig(BaseModel):
    provider: str = "openai"  # "openai", "anthropic"
    model_name: str = "gpt-4o"
    temperature: float = 0.7

class SearchConfig(BaseModel):
    enabled_sources: list[str] = ["valyu", "exa"]
    max_results_per_source: int = 10

class AgentConfig(BaseModel):
    num_agents: int = 5
    allow_subagents: bool = True
    max_search_rounds: int = 10

class AggregationConfig(BaseModel):
    aggregators: list[str] = ["mean", "supervisor"]

class CalibrationConfig(BaseModel):
    enabled: bool = False
    method: str = "platt"  # "platt", "extremise", "identity"
    calibration_data_path: str | None = None

class BacktestConfig(BaseModel):
    backtest_date: date
    foreknowledge_filter: bool = True

class PipelineConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    search: SearchConfig = SearchConfig()
    agents: AgentConfig = AgentConfig()
    aggregation: AggregationConfig = AggregationConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    backtest: BacktestConfig | None = None
```

---

## 7. Optimisations to Try

These are not part of the core system but are promising directions to explore once the baseline is working.

### 7.1 Walk-Forward Optimisation

Use backtesting results to iteratively improve the system:

1. System forecasts February events using only January data
2. Analyse what went wrong (missed information, flawed reasoning, data source gaps)
3. Generate targeted improvements (better prompts, additional data sources, domain-specific strategies)
4. System forecasts March events using February data + lessons learned
5. Repeat

This replicates the experience loop that makes human superforecasters good. Key risks: models may struggle to generate useful self-feedback, lessons from one time period may not generalise, and there's a risk of overfitting to the backtesting period. Should start with manual failure analysis before attempting to automate the feedback loop.

### 7.2 Distributional Forecasting

Extend beyond binary questions to full distributional outputs. For "when will X happen?", output a probability distribution over dates. For "what will GDP growth be?", output a distribution over values. Binary questions become CDF evaluations at a specific point. This is more general and more useful for decision-making, but harder to evaluate (need proper scoring rules for distributions, e.g. CRPS).

### 7.3 Consistency Checking

Run logical consistency checks across related forecasts. If P(A) = 0.7 and P(B) = 0.6, but A implies B, the system should flag the inconsistency. Similarly, probabilities across mutually exclusive events should sum to 1. This can be implemented as a post-hoc validation layer.

---

## 8. Key Technical Decisions

1. **Base model for live forecasting.** ✅ Model-agnostic via PydanticAI. Focusing on OpenAI and Anthropic models initially. Easy to swap as new frontier models are released.

2. **Search API selection.** ✅ Valyu (primary, backtestable) + Exa (secondary, partially backtestable). Additional sources (Tavily, NewsCatcher, NewsAPI.ai, Perigon) as future additions via the DataSource pattern (§1.2).

3. **Number of agents (M).** Start at 5, scale to 10. Diminishing returns beyond ~15. Cost scales linearly with M.

4. **Coupled vs. separated search+reasoning.** ✅ Coupled first (§1.3). Separated as a future configurable option.

5. **Aggregation strategy.** ✅ Mean + Supervisor to start (§2.2). TrimmedMean and ExtremisedMean as future additions. Track which aggregator performs best per domain.

6. **Evaluation dataset composition.** Starting with Metaculus (~4,000+ resolved questions). ForecastBench and Polymarket as additional sources. Sampling strategy across domains still needs work (§3.4).

7. **Calibration data bootstrapping.** Need ~200 resolved forecasts to fit calibration. Bootstrap from retroactive evaluation on historical benchmark questions.

8. **RL decision.** Deferred. Focus on scaffolding with frontier models first. Revisit when/if retrieval-in-the-loop RL becomes practical (§4.1).