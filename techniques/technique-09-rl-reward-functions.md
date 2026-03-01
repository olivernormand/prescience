# Technique 9: RL Reward Functions

**Category:** RL / training-dependent (requires model weight changes)
**Prescience stage:** Model training (§4)

## Overview

If we pursue RL fine-tuning, the choice of reward function matters enormously. Standard RL for language models rewards binary correctness ("did you get the right answer?"), but forecasting needs something more nuanced - we care about *how confident* the model was, not just whether it was right. These are three reward functions specifically designed for training calibrated forecasters.

**Context from §4.1:** The default bet is to use frontier models with good scaffolding. RL makes sense only under specific conditions (retrieval in the loop, ability to train on top of frontier models without delay). The techniques below are relevant if and when we decide to pursue RL.

---

### 9a. RLCR (Calibration Rewards) - Most Promising

Standard RL for language models (RLVR) rewards correct answers. But this actually *degrades* calibration - the model learns to be confident about everything, including things it's wrong about. RLCR fixes this by adding a calibration penalty:

**Reward = Correctness - (Confidence - Correctness)²**

The model generates a reasoning chain, a forecast, and a confidence estimate. The reward combines getting the answer right with getting the confidence right. Models trained this way learn to explicitly reason about their own uncertainty in the chain-of-thought - "I'm not sure about this because..." becomes a trained behaviour, not just a prompting artefact.

**Results (MIT, July 2025):** Calibration error reduced by up to 90% compared to standard RLVR, with no accuracy loss. Actually improved accuracy on out-of-distribution tasks, because well-calibrated uncertainty enables better reasoning under ambiguity.

**But wait - isn't this just Brier score?** Almost. If the reward is just the Brier score, the model is already being trained to minimise Brier. The RLCR distinction is that it explicitly separates correctness from calibration in the reward, which changes the learning dynamics. With pure Brier, the model can improve its score by being slightly more correct *or* slightly better calibrated - it doesn't know which lever it's pulling. RLCR makes the calibration lever explicit, which causes the model to learn to reason about its own uncertainty in the chain-of-thought. Whether this distinction matters in practice depends on the model and the training setup.

**Why it's the most promising for Prescience:** It trains explicit uncertainty reasoning - "I'm not sure about this because..." becomes a learned behaviour, not just a prompting artefact.

---

### 9b. Log Scoring Rule

Use the logarithmic scoring rule directly as the RL reward:

**Reward = log(p) if the event occurs, log(1-p) if it doesn't**

This is a "proper scoring rule" - mathematically, the model maximises expected reward by stating its true belief. The key property: assigning 0% to an event that occurs gives -∞ reward. This prevents the extreme-prediction collapse seen in some RL setups where models learn to always predict 0% or 100%.

**Results (March 2025):** Broke overconfidence patterns - models learned to use the full 0-100% range instead of clustering at the extremes. Generalised to unseen tasks without retraining.

**Comparison with 9a:** RLCR adds calibration as a penalty to correctness. Log scoring directly optimises a proper scoring rule. Log scoring is also theoretically compatible with logarithmic pooling (Technique 2b) - if you train agents on log score and aggregate via log pooling, the whole system optimises a consistent objective.

---

### 9c. Market Prices as Continuous Reward

Instead of waiting months for events to resolve (binary 0/1), train against prediction market prices which update daily. The model's forecast is scored against the current market price as a "soft label."

**Why this is attractive:** It solves the fundamental sample efficiency problem in forecasting RL. Binary outcomes are sparse (one signal per question, after months of waiting) and noisy (a 60% event still fails 40% of the time). Market prices provide dense, continuous signal.

**Why to be cautious:** Training directly against market prices will at best teach the model to *match* the market, not beat it. The reward signal says "you're right when you agree with the market" - so the model converges toward the market price. To outperform markets, you need a training signal that rewards being right when the market is wrong, which requires waiting for resolution (bringing you back to the sparse-reward problem).

Additionally: market prices are not ground truth, thin markets may have unreliable prices, and there's a circularity risk if market participants are themselves using LLM forecasts.

**Status:** Proposed but not yet validated (Lee et al., July 2025). The ceiling problem (can only match markets, not beat them) is a fundamental limitation that makes this less exciting than it initially appears.

---

### 9d. Behavioral Calibration RL (Claim-Level Confidence)

Two innovations from arXiv:2512.19920 (2025): using the PPO critic network's value function as an implicit confidence estimator (since the critic minimises Brier score between predicted value and policy return, it naturally converges to probability of success), and extending to **claim-level confidence** - different confidence scores for different sub-components of reasoning. It also uses Beta distribution priors for improved stability during GRPO training.

**Why this is interesting for forecasting:** A model could express high confidence in its base rate analysis but low confidence in a specific geopolitical factor assessment. This granularity goes beyond a single probability output and could help identify which parts of the reasoning are most uncertain.

---

### Research gap: RLAIF/Constitutional AI for Forecasting

Lee et al. (2025) identify the largest open research gap: no one has implemented RLAIF (Reinforcement Learning from AI Feedback) specifically for forecasting. The idea: train judge LLMs on expert annotations comparing forecasting reasoning quality, encoding a "forecasting constitution" with principles like: "Does the reasoning consider base rates?", "Does it weigh evidence for and against?", "Are probability estimates consistent with the reasoning?" This could provide richer reward signal than scalar Brier scores without waiting for event resolution.

---

## What to Measure

If pursuing RL:
- Compare RLCR, log scoring, and Brier-score-based rewards on calibration (ECE) and accuracy (Brier score)
- Check for extreme-prediction collapse (does the model use the full probability range?)
- Test whether agents trained with these rewards improve the overall system when plugged into the Prescience pipeline
