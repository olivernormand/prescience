# Technique 9: RL Reward Functions

**Category:** RL / training-dependent (requires model weight changes)
**Prescience stage:** Model training (§4)

## Overview

If we pursue RL fine-tuning, the choice of reward function matters enormously. Standard RL for language models rewards binary correctness ("did you get the right answer?"), but forecasting needs something more nuanced - we care about *how confident* the model was, not just whether it was right. The core choice is between proper scoring rules (Brier vs log score), with market prices as an alternative signal source.

**Context from §4.1:** The default bet is to use frontier models with good scaffolding. RL makes sense only under specific conditions (retrieval in the loop, ability to train on top of frontier models without delay). The techniques below are relevant if and when we decide to pursue RL.

---

### 9a. Brier Score as Reward (Default) / RLCR

The most natural reward function for forecasting: use the Brier score directly.

**Reward = 1 - (p - outcome)²**

The model outputs a probability, the event resolves, and the reward is how close the probability was to the outcome. This is a proper scoring rule - the model maximises expected reward by stating its true belief.

**RLCR (Damani et al., MIT, July 2025; arXiv:2507.16806)** was originally presented as a distinct approach: **R = Correctness - (Confidence - Correctness)²**. It was designed for QA tasks where the model produces a separate *answer* and a separate *confidence in that answer*. The reward explicitly separates "did you get it right" from "did you know how confident to be", and the authors showed this trains explicit uncertainty reasoning in the chain-of-thought ("I'm not sure about this because..."). Results: calibration error reduced by up to 90% vs standard RLVR, no accuracy loss, improved OOD performance.

**But for forecasting, RLCR collapses to Brier.** In forecasting, the probability *is* the output - there's no separate "answer" and "confidence." The model outputs 72% and the event either happens or doesn't. The RLCR formula applied to this case is just the Brier score. The interesting finding from the RLCR paper - that models learn to reason about uncertainty in their chain-of-thought - should emerge naturally from any proper scoring rule reward, because the model needs to calibrate well to maximise reward.

**The real question is Brier vs log score (9b)**, not Brier vs RLCR.

---

### 9b. Log Scoring Rule

Use the logarithmic scoring rule directly as the RL reward:

**Reward = log(p) if the event occurs, log(1-p) if it doesn't**

This is a "proper scoring rule" - mathematically, the model maximises expected reward by stating its true belief. The key property: assigning 0% to an event that occurs gives -∞ reward. This prevents the extreme-prediction collapse seen in some RL setups where models learn to always predict 0% or 100%.

**Results (March 2025):** Broke overconfidence patterns - models learned to use the full 0-100% range instead of clustering at the extremes. Generalised to unseen tasks without retraining.

**Comparison with Brier (9a):** Both are proper scoring rules. The key difference is the shape of the penalty: Brier penalises quadratically (being wrong by 0.3 is 9x worse than being wrong by 0.1), while log score penalises logarithmically (being wrong by assigning 0% to an event that occurs is *infinitely* bad). This means log score more aggressively punishes extreme overconfidence. Log scoring is also theoretically compatible with logarithmic pooling (Technique 2b) - if you train agents on log score and aggregate via log pooling, the whole system optimises a consistent objective.

---

### 9c. Market Prices as Continuous Reward

Instead of waiting months for events to resolve (binary 0/1), train against prediction market prices which update daily. The model's forecast is scored against the current market price as a "soft label."

**Why this is attractive:** It solves the fundamental sample efficiency problem in forecasting RL. Binary outcomes are sparse (one signal per question, after months of waiting) and noisy (a 60% event still fails 40% of the time). Market prices provide dense, continuous signal.

**Why to be cautious:** Training directly against market prices will at best teach the model to *match* the market, not beat it. The reward signal says "you're right when you agree with the market" - so the model converges toward the market price. To outperform markets, you need a training signal that rewards being right when the market is wrong, which requires waiting for resolution (bringing you back to the sparse-reward problem).

Additionally: market prices are not ground truth, thin markets may have unreliable prices, and there's a circularity risk if market participants are themselves using LLM forecasts.

**Status:** Proposed but not yet validated (Lee et al., July 2025). The ceiling problem (can only match markets, not beat them) is a fundamental limitation that makes this less exciting than it initially appears.

---

### 9d. Behavioral Calibration RL (Claim-Level Confidence)

From arXiv:2512.19920 (2025). The interesting idea for forecasting: instead of a single confidence score for the whole forecast, train the model to express **different confidence levels for different parts of its reasoning**. "I'm fairly sure the base rate for coups in this region is ~5% per year (high confidence), but I'm much less sure whether the current military tensions are different enough from baseline to warrant a large adjustment (low confidence)."

**Why this matters:** When a forecast turns out wrong, claim-level confidence tells you *which part* of the reasoning failed. Was the base rate wrong, or was the situation-specific adjustment wrong? This maps directly to BIN diagnostics (Technique 6a) - it's essentially building the diagnostic decomposition into the model's output rather than computing it after the fact. Uses Beta distribution priors for training stability and the PPO critic as an implicit confidence estimator.

---

### Research gap: RLAIF/Constitutional AI for Forecasting

Lee et al. (2025) identify the largest open research gap: no one has implemented RLAIF (Reinforcement Learning from AI Feedback) specifically for forecasting. The Brier score tells you *how wrong* a forecast was, but not *why* - a forecast can get a bad Brier score because the reasoning was sloppy, or because the event was genuinely unpredictable. RLAIF would train a judge model that evaluates the *quality of forecasting reasoning* independent of the outcome. A "forecasting constitution" could encode principles like: Did the reasoning start from an appropriate base rate? Did it weigh evidence for and against? Are the probability estimates consistent with the stated reasoning? Did it consider alternative scenarios?

This would provide reward signal that's richer than a scalar Brier score, available immediately (no waiting for resolution), and focused on reasoning quality rather than outcome luck.

---

## What to Measure

If pursuing RL:
- Compare Brier vs log scoring rewards on calibration (ECE) and accuracy (Brier score)
- Check for extreme-prediction collapse (does the model use the full probability range?)
- Test whether agents trained with these rewards improve the overall system when plugged into the Prescience pipeline
