# Technique 5: Validation & Quality Checks

**Category:** Inference-time (no model weight changes)
**Prescience stage:** Post-aggregation

## Overview

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

**Conformal prediction extension:** For a more rigorous version, use conformal prediction to produce intervals with statistical coverage guarantees. Compute a nonconformity score (e.g., deviation from the median prediction across runs), calibrate the threshold using a holdout set of resolved questions. The resulting interval has a provable coverage guarantee: if you set 90% coverage, at least 90% of future intervals will contain the well-calibrated probability. The MUSE framework (EMNLP 2025) uses Jensen-Shannon divergence to identify well-calibrated subsets of LLMs from a pool, then aggregates their outputs for tighter intervals. Key variants include Conformal Language Modeling (Quach et al., ICLR 2024, Google Research) and the "API Is Enough" method (Su et al., EMNLP 2024) for black-box LLMs without logit access.

**Triage value:** Unstable questions (wide spread) probably need more research or a different approach. Stable questions (narrow spread) can be output with less review.

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

**Complementary diagnostics:** The Martingale Score (arXiv:2512.02914, 2025; detects when models fail to properly update beliefs - predicts ground-truth accuracy without labelled data) and Bayesian Coherence Coefficient (arXiv:2507.17951, 2025; measures Bayesian consistency, r=0.906 correlation with model size) can serve as additional features for the meta-classifier or as standalone quality checks.

---

## What to Measure

- How often do raw forecasts violate consistency constraints (5a)? Does correction improve Brier scores?
- Does stability (5b) predict actual forecast error?
- Does meta-prediction (5c) successfully identify which forecasts are likely to be wrong?
- Does routing low-confidence forecasts to extended processing improve overall accuracy?
