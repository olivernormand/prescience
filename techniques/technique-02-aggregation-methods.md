# Technique 2: Aggregation Methods

**Category:** Inference-time (no model weight changes)
**Prescience stage:** Aggregation (§2.2)

## Overview

After M research agents each produce a probability estimate, we need to combine them into one (or more) final numbers. This is the aggregation step, implemented as pluggable `Aggregator` classes in the pipeline. The README already defines the architecture - this document covers the specific methods and their tradeoffs.

The honest starting point: simple averaging is surprisingly hard to beat. The diversity of search paths across agents does most of the work. Fancier aggregation methods add value only when the agents are meaningfully different from each other.

---

### 2a. Simple Mean (Baseline)

Average the M probabilities. That's it.

This is the baseline everything else must beat. It's robust, requires no tuning, and works well when agents are roughly similar in quality. The AIA Forecaster uses a variant of this (with extremisation applied afterward via Platt scaling).

---

### 2b. Logarithmic Pooling (Geometric Mean of Odds)

Instead of averaging probabilities directly, convert to log-odds, average, convert back. This handles extreme probabilities better - if three agents say 90%, 85%, 92%, log pooling gives a more confident result than arithmetic mean.

**Evidence:** Satopää et al. (2014) found log pooling with extremisation outperformed arithmetic mean in geopolitical forecasting, with optimal extremising factor d in [1.161, 3.921]. The Good Judgment Project used an "elitist extremising algorithm" - weighted by track record and update frequency, then extremised. Neyman & Roughgarden (Operations Research, 2023) proved a formal correspondence between scoring rules and aggregation methods - log scoring implies log pooling, quadratic (Brier) implies linear pooling. Since we score on Brier, the theoretical case for log pooling is actually weaker than it might seem.

**Extremisation detail:** Recent analysis suggests principled extremisation should push away from the estimated base rate, not from 50%. Using "pseudo-historical" odds as the baseline works better than assuming 1:1 odds. For LLM ensembles specifically, shared training data means arithmetic averaging will be systematically underconfident, making some form of extremisation necessary - but the extremising factor must be empirically tuned on held-out resolved questions.

**The concern:** Are we overfitting on statistics rather than making the models better forecasters? The aggregation formula is a finishing touch, not the main event. If log pooling beats simple mean on the backtest, great - use it. If it doesn't, the added complexity isn't worth it. Not something to refine heavily at this stage - the primary effort should go into making the agents better researchers and reasoners.

**Note:** The theoretical case for log pooling assumes somewhat independent forecasters. Our agents share training data, so the independence assumption is weak. This may reduce the benefit over simple averaging.

---

### 2c. Ensemble Configuration Selection

When we have meaningfully different agent configurations (different reasoning frameworks, different search strategies, different models), some combinations work better together than others. A mediocre approach that makes *different kinds of mistakes* from the others is more valuable to the ensemble than a strong approach that's correlated with everything else.

**This only matters when agents differ.** If all M agents are the same model with the same prompts, there's no stable difference between them - you can't meaningfully "select" a subset. The interesting part is giving them deliberately distinct strategies.

**Distinct strategies, not just random variation.** Not every agent needs the same "be a good superforecaster" prompt. We can assign different strategies and search for whether specific strategies are more performant in specific domains. For example:
- Agent A uses factor decomposition (Technique 4a) with a focus on base rates
- Agent B uses indicator-based search (Technique 3c) with structured hypothesis analysis
- Agent C uses dual-perspective retrieval (Technique 3b) with argue-both-sides reasoning
- Agent D focuses exclusively on prediction market and structured data sources

Each strategy is a loadable configuration. The question then becomes: which strategies work best for which domains? Indicator-based analysis might excel at geopolitics. Market-data-heavy agents might dominate economics. The system learns to weight strategies differently per domain.

**Evidence:** Fox et al. (2024, *Emerging Infectious Diseases*, CDC) found that selecting ensemble members by contribution to ensemble performance (not individual accuracy) improved skill by 6.1%, and size-4 ensembles matched the full CDC hub.

**Implementation:** After accumulating backtest results across strategy configurations and domains, evaluate which strategy combinations complement each other (i.e. make different kinds of mistakes) and select the best subset per domain.

---

### 2d. Supervisor Aggregator (LLM-Based Reconciliation)

Instead of a statistical formula, an LLM reads all M agent forecasts with their reasoning and produces a reconciled estimate. It can identify where agents disagree, do targeted follow-up search to resolve factual disputes, and check for systematic biases.

This is the most expensive aggregation method but potentially the most powerful - it can reason about *why* agents disagree, not just average their numbers. Defined in detail in the README (§2.2).

---

### 2e. Delphi (Iterative Deliberation)

Multiple rounds of anonymous peer feedback. After the initial forecasts, each agent sees the others' estimates and reasoning (anonymised), revises their own, and repeats for 3-4 rounds until convergence.

**Evidence:** Bertolotti & Mari (2025; arXiv:2502.21092) formalised the LLM-based Delphi method. Mueller et al. (2024) found moderate correlation (r = 0.64) between AI and human expert Delphi panels.

**Caveat:** A NeurIPS 2025 paper (disentangling debate from voting) found that majority voting alone accounts for most gains typically attributed to multi-agent debate. The iterative deliberation structure needs to genuinely add value beyond simple averaging. Worth testing, but be skeptical.

---

## What to Measure

Compare all aggregation methods on Brier score across the backtest suite. The key questions:
- Does anything consistently beat simple mean + Platt scaling?
- Does the supervisor aggregator's additional cost justify its improvement?
- Does domain-specific agent selection (2c) outperform using all agents uniformly?

Run multiple aggregators in parallel on the same question to get head-to-head comparisons.
