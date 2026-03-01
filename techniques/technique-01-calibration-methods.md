# Technique 1: Calibration Methods

**Category:** Inference-time (no model weight changes)
**Prescience stage:** Calibration (§2.3)

## Overview

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

Borrowed from weather forecasting (Gneiting et al., 2005). Instead of calibrating the final aggregated probability, EMOS calibrates the ensemble directly - fitting a parametric distribution whose parameters are linear functions of ensemble member forecasts. The mean is a bias-corrected weighted average of agent outputs; the variance is a linear function of ensemble spread. Parameters are estimated by minimising CRPS (Continuous Ranked Probability Score).

The key insight from meteorology is the **spread-skill relationship**: ensemble spread predicts forecast uncertainty. If agents disagree widely, the forecast is less reliable. EMOS formalises this - it learns to widen the uncertainty when agents disagree and narrow it when they agree. Bayesian Model Averaging (BMA) takes a similar approach, modelling the predictive distribution as a weighted mixture of PDFs centred on each ensemble member - BMA produced 90% prediction intervals 66% shorter than climatology baselines.

**Honest assessment:** This makes most sense when the ensemble members are meaningfully different (different models, different techniques). When our agents are M runs of the same model, the individual-level weighting probably doesn't buy much. The main value would be learning the relationship between agent disagreement and forecast error - but as discussed in Technique 1a, uncertainty about our estimate should ideally be baked into the probability itself, not expressed as a separate layer.

**When to use:** Once we have diverse agent configurations (different reasoning frameworks, different models). Probably not useful in the early days when all agents are similar.

---

### 1d. Venn-ABERS Calibration

Provides stronger theoretical guarantees than Platt scaling, especially with small calibration datasets. Where Platt assumes the calibration curve is logistic (which may not be true), Venn-ABERS (van der Laan et al., 2025; arXiv:2502.05676) extends Vovk's framework to provide **finite-sample calibration guarantees** - unlike isotonic regression, which requires asymptotic convergence. It applies isotonic regression twice (once assuming each class), producing a calibrated interval [p0, p1]. This outputs not just a calibrated point estimate but a calibrated *interval*, communicating uncertainty in the calibration itself.

**The inverse softmax trick** (Wang et al., 2024; arXiv:2410.06707) is a related technique worth noting: it inverts LLM-generated verbalized probabilities back to estimated logits, then applies temperature scaling. This enables post-hoc calibration of black-box LLMs that only output text. Essentially the same category as Platt scaling but approaches the problem from the other direction.

**When to use:** Early in the system's life when we have <100 resolved forecasts. Platt scaling can overfit with small datasets; Venn-ABERS is more robust. As we accumulate more data, the advantage shrinks and Platt's simplicity wins.

---

## What to Measure

Compare all calibration methods on Brier score across the backtest suite. The key questions:
- Does any method consistently beat Platt scaling?
- Does per-domain calibration (1b) outperform global calibration?
- Does the winner change as the calibration dataset grows?

Calibration should be a clean on/off toggle in the pipeline config so we can measure whether it helps at all for a given model + aggregator combination.
