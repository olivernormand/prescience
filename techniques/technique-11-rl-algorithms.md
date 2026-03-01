# Technique 11: RL Algorithms & Engineering

**Category:** RL / training-dependent (requires model weight changes)
**Prescience stage:** Model training (§4)

## Overview

Given a reward function (Technique 9) and training data (Technique 10), which RL algorithm should we actually use, and what practical engineering is needed to make it work? Forecasting RL has specific failure modes that general-purpose RL doesn't encounter.

---

### 11a. ReMax (Recommended Default)

ReMax with baseline-subtracted advantages outperformed all GRPO variants for forecasting in direct comparisons. A 7-run ensemble of ReMax-trained models achieved Brier score 0.190, beating OpenAI o1.

**Why ReMax over GRPO for forecasting specifically:** GRPO computes advantages relative to other samples in the same batch - "was this forecast better than the others in this batch?" But forecasting rewards are inherently noisy (a correct 60% forecast gets punished 40% of the time), so the batch-relative comparison is unstable. ReMax uses a learned baseline instead - "was this forecast better than what we'd normally expect?" - providing more stable gradient estimates when individual rewards are noisy.

**Practical engineering that matters:**
- **Soft Brier fallback:** When the model outputs a malformed forecast (not a valid probability), assign a constant 0.25 Brier score instead of crashing or giving 0 reward. This prevents training collapse over long runs
- **Guard-rails:** Token-length limits (prevent unbounded reasoning chains), gibberish filters (detect degenerate outputs), early-stop criteria (stop training when validation metrics plateau)
- **Extreme-prediction collapse:** A persistent problem with GRPO where models converge to always predicting 0% or 100%. ReMax with baseline subtraction mitigates this, but it needs monitoring

**Ensemble approach:** The best reported result (Brier 0.190) came from a 7-run ensemble - training the same model with 7 different random seeds and ensembling their outputs. This fits naturally with Prescience's agent ensemble: each agent could be a different ReMax-trained model seed.

---

### 11b. Curriculum Learning

Not all training questions are equally useful. Questions where the model has roughly a 50% chance of getting it right are the most informative (maximum gradient signal). Very easy questions (model already knows the answer) and very hard questions (essentially random) provide little learning signal.

**Prompt Curriculum Learning (PCL):** Train a value model that estimates question difficulty for the forecasting model, then schedule training from medium-difficulty to harder questions. Achieved 12-16x faster convergence on math benchmarks.

**For forecasting:** Questions near 50% on prediction markets are naturally "harder" and provide the most learning signal. Start training on these, then gradually include easier and harder questions.

**Practical implication:** Don't train on a random sample of all questions. Weight the training set toward questions in the 30-70% range where the model has the most to learn.

---

### 11c. GRIP (Group-based Relative Importance for Policy Optimization)

From Time-R1 (arXiv:2506.10630, 2025). The core idea applied to forecasting: not all training rollouts are equally informative. A forecast where the model agonised between 55% and 65% and landed on 60% teaches less than one where it initially estimated 30%, found contradicting evidence, and revised to 70%. GRIP selects the most informative trajectories from a larger candidate pool and amplifies their gradient signal, so the model learns disproportionately from the rollouts where its reasoning made the biggest difference.

---

### The Lee et al. framework: Three fundamental training challenges

Lee et al.'s position paper (July 2025; arXiv:2507.19477) deserves attention because it identifies structural problems in forecasting RL that current approaches only partially address:

1. **The noisiness-sparsity problem:** Event outcomes are inherently stochastic - a 60% probability event still fails 40% of the time - making reward signals extremely noisy compared to math or code where answers are deterministic.

2. **The simple reward structure problem:** Unlike math (multi-step verifiable reasoning where each step can be checked), forecasting gives you one number at the end - "your Brier score was 0.18." That tells you almost nothing about *which parts* of the reasoning were good or bad. Did the agent find the right evidence but weigh it wrong? Or miss key evidence entirely? Solutions include: LLM-as-judge for reasoning evaluation (Technique 9 research gap), decomposing questions into verifiable sub-questions (e.g., "What is SpaceX's recent track record?" can be checked independently of the final forecast), and using prediction market prices as continuous intermediate reward signals (Technique 9c, with caveats).

3. **The knowledge contamination problem:** Pre-cutoff events may be in the model's training data, meaning "forecasting" is actually recall. Techniques 10b and 10c address this directly.

These challenges explain why forecasting RL is harder than math/code RL and why the training data generation techniques (Technique 10) are as important as the algorithms themselves.

---

## What to Measure

- ReMax vs. GRPO on Brier score, calibration error, and training stability
- Does curriculum learning (starting with medium-difficulty questions) improve convergence speed?
- Does the 7-seed ensemble approach produce diverse enough models to benefit from ensembling?
- Monitor for extreme-prediction collapse throughout training
