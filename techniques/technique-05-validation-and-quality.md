# Technique 5: Validation & Quality Checks

**Category:** Inference-time (no model weight changes)
**Prescience stage:** Post-aggregation

## Overview

**Tetlock Commandment:** 6 (distinguish degrees of doubt → 5d)

After producing a forecast, run checks to catch errors before output. These are safety nets - they don't improve the forecast directly but catch cases where the system has made a mistake it could have detected.

**A prerequisite problem:** Current models show massive run-to-run variation. You can get substantially different probability estimates from the same model on the same question just by running it again. Until this is addressed (likely by running ~10 times and averaging to get something internally consistent), it's worth asking how useful any downstream validation really is. If the raw signal is noisy, sophisticated post-processing may be polishing noise. The most important "validation" might simply be ensembling enough runs to beat down the variance.

---

### 5a. Cross-Question Consistency

LLMs are bad at being logically consistent across related questions. If you ask "Will X happen by June?" and "Will X happen by December?", the second should always be at least as high as the first - but LLMs regularly violate this. The fix: forecast a family of related questions, check whether the answers are logically consistent, and if not, adjust them.

**The Q4 2024 Metaculus AI Benchmarking winner used exactly this** to improve their estimates. It's one of the more proven techniques available, and we already use similar self-consistency approaches in other areas.

**How it works:**
1. For each question, generate logically related questions (different time horizons, complementary outcomes, conditional variants)
2. Forecast each independently
3. Check for logical violations (probabilities that should be monotonic but aren't, complementary events that don't sum to 100%, etc.)
4. If violated, find the smallest adjustments that restore consistency

**The violations themselves are informative:** If the system consistently gives higher probabilities for shorter time horizons, that tells you something about a systematic reasoning failure worth investigating.

**Causal consistency extension:** Beyond strict logic, check whether causally linked forecasts are approximately consistent. If the system says "70% chance rates rise" and "80% chance inflation falls", and it believes rate rises cause inflation to fall, the conditional structure should roughly hang together.

**As an RL training signal:** Consistency violations could also serve as a reward signal for RL training (Technique 10). If the model produces inconsistent probabilities across related questions, that's a training signal that doesn't require waiting for events to resolve - similar in spirit to path-independence training (Technique 10a).

**Cost:** Multiplies the number of questions to forecast. Best for high-value questions or question families that arise naturally from the decomposition step.

---

### 5b. Stability Testing (Conformal Prediction)

Run the pipeline multiple times for the same question with minor variations (different seeds, different search result orderings) and look at how much the answers vary. If they cluster tightly, the system is internally consistent. If they're spread widely, something is fragile.

**What this tells you:** Stability, not accuracy. A narrow spread means the system reliably produces the same answer - but it could be reliably wrong (all agents agree for the wrong reasons due to shared training data and similar search results). Treat this as a diagnostic, not a confidence measure.

**Formalising this as confidence intervals:** Conformal prediction can turn this raw spread into intervals with statistical guarantees. If you calibrate the spread against historical forecast accuracy (using resolved backtest questions), you can say "we're 90% confident the true well-calibrated probability is between 55% and 75%." The width of that interval is itself the diagnostic: narrow means the system is confident, wide means it isn't. The MUSE framework (EMNLP 2025) extends this to multi-model settings - it identifies which of your agent configurations are well-calibrated for which question types, and weights them accordingly. Conformal Language Modeling (Quach et al., ICLR 2024) and the "API Is Enough" method (Su et al., EMNLP 2024) provide variants that work with black-box models.

**Triage value for the pipeline:** Unstable questions (wide spread) probably need more research or a different approach - route them to extended processing. Stable questions (narrow spread) can be output with less review. This is a natural fit for the compute allocation decisions in Technique 7.

**Cost:** Expensive - requires multiple full pipeline runs. Best used selectively during development to understand which question types produce fragile forecasts.

---

### 5c. Meta-Prediction (Predicting Forecast Quality)

Train a separate classifier that predicts whether each specific forecast is likely to be accurate, based on features of the reasoning, search results, and question characteristics. Use this to route low-confidence forecasts to additional processing.

**Features to use:**
- Agent agreement (how much did the M agents agree?)
- Search quality (how many sources found, how recent, how relevant?)
- Question characteristics (domain, time horizon, novelty)
- Reasoning features (how many evidence sources cited, how complex the reasoning)

**Triage routing:**
- High meta-confidence → output directly
- Low meta-confidence → trigger extended processing (more search, human review)

This concentrates expensive compute on exactly the questions that need it. The meta-classifier trains on resolved backtest forecasts using the features above.

**Evidence:** Schoenegger et al. (2023) demonstrated that ML models can predict a priori whether a human geopolitical forecast is likely to be accurate, using features from the written rationale (integrative complexity, hedging language, number of sources cited).

**Detecting when the model isn't actually updating on evidence:** Two recent diagnostics can help identify forecasts where the model ignored what it found during research:

- **Martingale Score** (arXiv:2512.02914, 2025): Checks whether the model's probability actually shifted in response to the evidence it found, or whether it just anchored on its prior and wrote a post-hoc justification. Belief entrenchment is widespread across models - and RL-trained reasoning models can be *worse* at updating than base models. This predicts ground-truth accuracy without needing resolved outcomes.
- **Bayesian Coherence Coefficient** (arXiv:2507.17951, 2025): Measures whether the model updates its beliefs in a way consistent with Bayes' theorem. All models systematically under-update (r=0.906 correlation between model size and coherence - larger models are closer to Bayesian but still fall short).

Both could serve as features for the meta-classifier, or as standalone red flags: if the model's probability barely moved despite finding strong evidence, something is wrong with that forecast.

---

### 5d. Probability Granularity Discipline (Tetlock Commandment 6)

Tetlock's sixth commandment: "Strive to distinguish as many degrees of doubt as the problem permits." GJP data showed that superforecasters achieved calibration of 0.01 (1 percentage point average error), and that accuracy *degraded* when predictions were rounded to the nearest 10%. The granularity of probability estimates correlates with accuracy - forecasters who distinguish 60% from 65% outperform those who cluster on round numbers.

**The problem for LLMs:** Models tend to cluster on round numbers (50%, 70%, 80%, 90%). This is a known failure mode - the model reaches for a "feels about right" round number rather than reasoning precisely about where the evidence places the probability. A forecast of 70% when the evidence actually supports 64% is throwing away signal.

**Implementation as a validation check:**
1. After the agent produces a probability, check whether it falls on a round number (multiples of 10, or even 5)
2. If it does, prompt the agent to reconsider: "You estimated 70%. Is the evidence more consistent with 65-69% or 71-75%? Be precise."
3. Track round-number frequency across runs - a system that produces round numbers >50% of the time is being imprecise

**Implementation as a prompt directive:** Include in the agent's reasoning prompt: "State your probability to at least the nearest 5%. If the evidence supports distinguishing finer (e.g. 62% vs 67%), do so. Avoid defaulting to round numbers - there is a real difference between 60% and 65%."

**Why this matters beyond aesthetics:** Granularity is a proxy for the depth of reasoning. A forecaster who says "about 70%" hasn't thought as carefully as one who says "68% - the base rate is 60%, and the two strong pieces of evidence I found each push it up by about 4 points." The precision forces a more rigorous accounting of how each piece of evidence shifts the probability.

**Caveat:** False precision is worse than honest imprecision. The goal isn't to force 1% granularity on every question - it's to avoid lazy rounding when the evidence supports more precision. For genuinely uncertain questions (forecastability gate, Technique 7d), reporting "roughly 50-55%" is more honest than a spuriously precise "52%."

---

## What to Measure

- How often do raw forecasts violate consistency constraints (5a)? Does correction improve Brier scores?
- Does stability (5b) predict actual forecast error?
- Does meta-prediction (5c) successfully identify which forecasts are likely to be wrong?
- Does routing low-confidence forecasts to extended processing improve overall accuracy?
- What fraction of agent forecasts fall on round numbers (5d)? Does prompting for granularity improve Brier scores?
