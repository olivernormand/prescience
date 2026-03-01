# Technique 2: Aggregation Methods

**Category:** Inference-time (no model weight changes)
**Prescience stage:** Aggregation (§2.2)

## Overview

**Tetlock Commandments:** 9 (bring out the best in others → 2f)

After M research agents each produce a probability estimate, we need to combine them into one (or more) final numbers. This is the aggregation step, implemented as pluggable `Aggregator` classes in the pipeline. The README already defines the architecture - this document covers the specific methods and their tradeoffs.

The honest starting point: simple averaging is surprisingly hard to beat. The diversity of search paths across agents does most of the work. Fancier aggregation methods add value only when the agents are meaningfully different from each other.

---

### 2a. Simple Mean (Baseline)

Average the M probabilities. That's it.

This is the baseline everything else must beat. It's robust, requires no tuning, and works well when agents are roughly similar in quality. The AIA Forecaster uses a variant of this (with extremisation applied afterward via Platt scaling).

---

### 2b. Logarithmic Pooling (Geometric Mean of Odds)

Instead of averaging probabilities directly, convert to log-odds, average, convert back. The practical difference shows up when agents are confidently agreeing: if three agents say 90%, 85%, 92%, arithmetic mean gives ~89% but log pooling gives ~91%. Log pooling respects the idea that when multiple independent assessments all point toward an extreme probability, the combined estimate should be more extreme, not just the average.

**Evidence from forecasting:** Satopää et al. (2014) found log pooling with extremisation outperformed arithmetic mean in the Good Judgment Project's geopolitical forecasting, with optimal extremising factor d in [1.161, 3.921]. The GJP used an "elitist extremising algorithm" - weighting forecasters by track record and update frequency, then extremising. Neyman & Roughgarden (Operations Research, 2023) proved a formal correspondence: log scoring implies log pooling, Brier scoring implies linear pooling. Since we score on Brier, the theoretical case for log pooling is actually weaker than it might seem.

**Why extremisation matters for us specifically:** Our agents share the same underlying model and training data, so they hedge in correlated ways. If five agents independently research a question and all come back saying 70%, that's a stronger signal than one agent saying 70% - but arithmetic mean just returns 70% regardless. Extremisation corrects for this shared hedging bias. Recent analysis suggests the push should be away from the base rate, not from 50% - if the base rate for this class of events is 40% and our agents say 70%, that 30-point deviation from the base rate should be amplified. The extremising factor must be tuned on held-out resolved forecasts.

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

Mimics how the best human forecasting teams actually work: after making initial forecasts, each agent sees the others' estimates and reasoning (anonymised), revises their own, and repeats for 3-4 rounds until convergence. The idea is that agents can identify holes in each other's reasoning - "you didn't consider that the election commission has already set a date" - and update accordingly.

**Evidence:** Bertolotti & Mari (2025; arXiv:2502.21092) formalised the LLM-based Delphi method. Mueller et al. (2024) found moderate correlation (r = 0.64) between AI and human expert Delphi panels on forecasting tasks.

**Caveat:** A NeurIPS 2025 paper (disentangling debate from voting) found that majority voting alone accounts for most gains typically attributed to multi-agent debate. The deliberation might not add much beyond what simple averaging already captures. The test: do the forecasts meaningfully change during deliberation rounds, or do they just converge to the mean? If the latter, we're paying for multiple rounds to get the same result as averaging.

---

### 2f. Cooperative Team Forecasting (GJP-Inspired)

Inspired by how the best Good Judgment Project teams operate. The key insight from GJP research: top-performing teams weren't just collections of individually strong forecasters - they were groups that *worked well together*. The diversity of cognitive styles within the team reduced correlated errors, and the cooperative dynamic meant that bad reasoning got challenged before it infected the final forecast.

**How this differs from Delphi (2e):** Delphi is structured and anonymous - agents see numbers and reasoning, revise in isolation, repeat. The GJP team model is more organic: agents have distinct personalities and cognitive styles, and they *argue cooperatively*. The goal isn't convergence through averaging rounds - it's collective intelligence through productive disagreement.

**How this differs from ensemble configuration (2c):** In 2c, agents have different *strategies* (different search approaches, different reasoning frameworks). Here, agents all have the same goal and the same problem, but different *personalities* - some are naturally more disagreeable, some more cautious, some more willing to take contrarian positions. The diversity comes from cognitive style, not task assignment.

**The mechanism:**
1. All M agents independently research and forecast the same question (as in the current architecture)
2. Agents are assigned distinct personality profiles - not different research strategies, but different cognitive dispositions. Examples:
   - A "challenger" personality that's naturally sceptical and pushes back on weak reasoning
   - A "synthesiser" that looks for common ground and tries to reconcile disagreements
   - An "outside-view anchor" that stubbornly returns to base rates when others get swept up in narratives
   - A "detail-oriented" personality that fixates on specific evidence quality and source reliability
   - A "contrarian" that's disposed to ask "what if everyone here is wrong?"
3. After initial independent forecasts, agents enter a **cooperative deliberation** phase: they share findings, challenge each other's reasoning, and argue - but cooperatively, not adversarially. The goal is to *improve the group's forecast*, not to win the argument
4. The group works toward a shared forecast through productive disagreement

**Why personality diversity matters:** If all agents think the same way, they make the same mistakes. The evidence from GJP is that diversity of *thinking style* is what drives forecast quality - it reduces the probability that a bad forecast survives unchallenged. A team where everyone agrees easily produces confident, correlated errors. A team where someone always asks "but what about the base rate?" or "are we anchoring on that one news article?" catches mistakes that homogeneous groups miss.

**Key distinction from adversarial approaches (Technique 10d):** This isn't debate. The agents aren't trying to defeat each other's arguments. They're trying to arrive at the best possible collective forecast. The "disagreeable" personality isn't a devil's advocate performing a role - it's an agent that genuinely has a lower threshold for accepting weak reasoning. The dynamic is a team that disagrees agreeably.

**Implementation considerations:**
- Personality profiles are prompt modules that shape the agent's disposition, not its research strategy or reasoning framework (those come from Techniques 3 and 4). An agent can be "disagreeable" while using factor decomposition (4a) or Bayesian updating (4e)
- The deliberation phase needs to be carefully designed to avoid the NeurIPS 2025 finding that debate collapses to majority voting. The key is whether agents actually *change their reasoning* in response to challenges, not just shift their numbers. Track whether the reasoning chains substantively update between rounds
- The number of deliberation rounds should be adaptive: stop when reasoning has stabilised, not after a fixed number of rounds
- Personality diversity should be genuinely diverse, not just variations on "be more confident" vs "be less confident." The GJP research emphasises cognitive diversity - different ways of *approaching* problems, not just different calibration levels

**Evidence:** Mellers et al. (2014, *Psychological Science*) found that GJP teams outperformed prediction markets by 15-20% on accuracy. The team effect was additive to individual skill - good forecasters in good teams beat good forecasters working alone. Woolley et al. (2010, *Science*) identified "collective intelligence" as a measurable property of groups that's distinct from individual intelligence, driven by social sensitivity, conversational turn-taking, and (importantly) cognitive diversity.

**Open question:** How much of the GJP team benefit translates to LLM agents? Human teams benefit from genuine knowledge differences (one person knows about economics, another about politics). LLM agents share the same training data, so the "knowledge diversity" channel is weaker. The bet here is that *reasoning style diversity* (prompted through personality profiles) provides enough decorrelation to improve forecasts. This is testable through the backtest loop.

---

## What to Measure

Compare all aggregation methods on Brier score across the backtest suite. The key questions:
- Does anything consistently beat simple mean + Platt scaling?
- Does the supervisor aggregator's additional cost justify its improvement?
- Does domain-specific agent selection (2c) outperform using all agents uniformly?
- Does cooperative team deliberation (2f) outperform both simple averaging and structured Delphi? Specifically: do forecasts meaningfully improve during deliberation, or just converge to the prior mean?
- Does personality diversity (2f) reduce correlated errors compared to M copies of the same persona?

Run multiple aggregators in parallel on the same question to get head-to-head comparisons.
