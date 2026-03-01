# Technique 1: Calibration Methods

**Category:** Inference-time (no model weight changes)
**Prescience stage:** Calibration (§2.3)

## Overview

**Tetlock Commandment:** 7 (balance under- and overconfidence)

After the agents have done their research and the aggregation step has combined their outputs into a single probability, we may want to apply a final correction. LLMs systematically hedge toward 50% - if the evidence says 80%, the system might output 68%. Calibration learns this pattern from historical data and corrects for it.

All calibration methods share the same basic structure: take the raw probability, look up what the actual outcome frequency was when the system said that probability in the past, and adjust accordingly. The differences are in how they fit that correction curve and what guarantees they provide.

**The AIA Forecaster evaluated most of these and chose Platt scaling.** That's a strong signal. Start there.

---

### 1a. Platt Scaling with Extremisation (Default)

The simplest and most proven approach. Fit a logistic curve through historical forecast-outcome pairs:

P(true) = 1 / (1 + e^(A·logit(p) + B))

Two parameters (A and B), learned from backtest data. When A > 1, this pushes probabilities away from 50% (extremisation). When B ≠ 0, it corrects for a directional bias (e.g. the system consistently overestimates).

**Extremisation** deserves special mention: because our agents share training data, their "independent" estimates aren't truly independent. They all hedge in similar ways. Extremisation says "if multiple agents agree on 70%, the true answer is probably more like 75-80%." A common factor is α=√3 ≈ 1.73, but this should be tuned on backtest data.

**Requirements:** ~100-200 resolved forecasts to fit reliably. Bootstrap from retroactive evaluation on Metaculus/Polymarket/ForecastBench questions.

**This is the default. Everything below is a variant to consider if Platt doesn't work well enough.**

---

### 1b. Per-Domain Calibration (Thermometer)

The same idea as Platt scaling, but fit separate parameters for different question types. Political questions might need aggressive extremisation (A=1.8) while economic questions might need less (A=1.2). This comes from the MIT/IBM "Thermometer" paper (ICML 2024) which showed that learning per-category calibration parameters outperforms a single global temperature.

**When to use:** If backtest analysis shows the system is miscalibrated in different directions for different domains (e.g. overconfident on geopolitics but underconfident on economics), per-domain calibration will help. If the miscalibration is roughly uniform across domains, it's not worth the complexity.

**Risk:** With per-domain calibration you're fitting more parameters on less data (splitting your calibration set by domain). Need enough resolved questions per domain to avoid overfitting.

---

### 1c. EMOS (Ensemble Model Output Statistics)

Borrowed from weather forecasting (Gneiting et al., 2005), where the same problem exists: you have an ensemble of models, each producing a forecast, and you need to learn how to combine and correct them. EMOS learns a formula that does two things at once: it learns which agents to trust more (a weighted average of their outputs), and it learns the relationship between how much the agents disagree and how wrong the final forecast tends to be.

**The forecasting-specific insight:** When our 5-10 agents produce estimates of 60%, 65%, 70%, 72%, 75% - that tight clustering means something different than if they produce 30%, 50%, 65%, 80%, 90%. EMOS learns from historical forecasts what each spread pattern means for actual accuracy. Weather forecasters call this the "spread-skill relationship" - wide spread predicts larger errors. BMA (Bayesian Model Averaging) is a related approach that produced 90% prediction intervals 66% shorter than baselines in weather.

**Honest assessment:** This makes most sense when the ensemble members are meaningfully different - different search strategies, different reasoning frameworks, different models. When our agents are M runs of the same model with the same prompts, the spread is mostly random noise and there's less stable signal to learn from. The main value comes once we have genuinely diverse agent configurations (Technique 2c).

**When to use:** Once we have diverse agent configurations and enough resolved forecasts to learn the spread-error relationship reliably. Probably not useful in the early days when all agents are similar.

---

### 1d. Venn-ABERS Calibration

The practical problem: early in Prescience's life, we'll have very few resolved forecasts to calibrate against - maybe 30 or 50. Platt scaling can overfit badly on small datasets (fitting a logistic curve through 30 points is unreliable). Venn-ABERS (van der Laan et al., 2025; arXiv:2502.05676) is more robust in this regime - it provides finite-sample calibration guarantees rather than assuming a specific curve shape. It also outputs a calibrated *interval* [p0, p1] rather than a point, which tells you how uncertain the calibration itself is - useful when you're still learning the system's biases.

**The inverse softmax trick** (Wang et al., 2024; arXiv:2410.06707) solves a different practical problem: if we're using a black-box model that only gives us text probabilities (no logits), how do we calibrate it? The trick inverts the verbalized probability back to an estimated logit, then applies temperature scaling. Relevant if we're calibrating outputs from closed-source models.

**When to use:** Early in the system's life when we have <100 resolved forecasts. As we accumulate more data, Platt becomes reliable and its simplicity wins.

---

## What to Measure

Compare all calibration methods on Brier score across the backtest suite. The key questions:
- Does any method consistently beat Platt scaling?
- Does per-domain calibration (1b) outperform global calibration?
- Does the winner change as the calibration dataset grows?

Calibration should be a clean on/off toggle in the pipeline config so we can measure whether it helps at all for a given model + aggregator combination.
