# Technique 10: RL Training Data Generation

**Category:** RL / training-dependent (requires model weight changes)
**Prescience stage:** Model training (§4)

## Overview

The hardest part of RL for forecasting isn't the algorithm - it's getting enough training data. Events take months to resolve, outcomes are noisy (a 60% event fails 40% of the time), and each resolved question gives only a single scalar reward. These techniques generate more (or better) training signal from limited data.

---

### 10a. Path-Independence Training (Self-Supervised)

A well-calibrated forecaster should produce the same probability regardless of the order it receives information. Show it facts A, B, C simultaneously and it should give the same answer as A→B→C or C→A→B. If it doesn't, it's being irrational - the order shouldn't matter.

**Why this is powerful:** It generates unlimited training signal without waiting for events to resolve. Take any historical period, sample different orderings of the available information, and penalise the model for giving different answers on different orderings. One period with N information items generates many possible orderings to train on.

**Theoretical grounding:** De Finetti's Dutch book theorem proves that path-dependent probabilities are exploitable - there's a set of bets that guarantees a loss against a path-dependent forecaster. So this isn't just a nice-to-have; it's a necessary condition for rational probability assignment.

**Why it might not work in practice:** Real forecasting often involves sequential updates where the *framing* of new information (not just its content) legitimately affects interpretation. "GDP dropped 2%" means different things if you already know about a recession versus if it's the first bad signal. Still, large path-dependence is almost certainly a bug.

---

### 10b. Counterfactual Event Training

Generate plausible hypothetical questions about events that didn't happen. "What if SpaceX's test had failed?" "What if the election had been delayed?" The model must reason about the counterfactual scenario using its understanding of causal structure, not recall the actual outcome.

**Why this helps:** Massively expands the training dataset beyond the limited set of real resolved questions. Also forces genuine reasoning - the model can't just recall what happened, because the counterfactual didn't happen.

**Implementation:** Use an LLM to generate plausible alternative scenarios from historical events. For each counterfactual, define a resolution criterion and a plausible outcome. Train the model to forecast the counterfactual event.

**Risk:** The quality of the training signal depends on the quality of the counterfactual generation. Implausible or poorly-defined counterfactuals will generate noise, not signal.

**Hypothetical Event Bayesian Networks:** An extension proposed by Lee et al. (July 2025; arXiv:2507.19477) - construct Bayesian networks of related events to model hidden inter-event correlations, providing richer reward signal. This directly addresses the **noisiness-sparsity problem**: event outcomes are inherently stochastic (a 60% probability event still fails 40% of the time), making individual reward signals extremely noisy compared to math or code where answers are deterministic. By modelling correlations between events, you can extract denser reward signal from the same data.

---

### 10c. Poorly-Recalled Events Curriculum

Find pre-cutoff events that the model doesn't remember well (verified by probing its knowledge). Train on these, because the model genuinely has to forecast rather than recall.

**Why this helps:** Avoids the knowledge-cutoff contamination problem that plagues backtest-based training. If the model already knows the answer (even implicitly from training data), the training signal is about recall, not forecasting skill. Poorly-recalled events provide a cleaner signal.

**Implementation:** Before training, probe the model's knowledge of candidate events. Select events where the model shows genuine uncertainty about the outcome. These become the training set.

---

### 10d. Adversarial Self-Play

Two model instances play a structured game: a forecaster produces a prediction with reasoning, and a challenger tries to find flaws - counter-evidence, ignored base rates, implicit conjunctions. The game outcome provides training signal.

**Specialised challengers:**
- **Base-rate enforcer:** Checks whether the forecast is justified given historical frequencies
- **Devil's advocate:** Constructs the strongest case for the opposite outcome
- **Conjunction auditor:** Flags implicit conjunctions ("X will happen because A, B, and C all occur" - what's the joint probability of A, B, and C?)

**Can also be used at inference time** (without training): deploy challengers as part of the agent pipeline to stress-test reasoning before finalising a forecast.

**Caveat:** A NeurIPS 2025 finding suggests majority voting accounts for most gains attributed to multi-agent debate. The adversarial structure needs to be specifically designed to add value beyond simple ensembling - the specialised challengers above are an attempt to do this.

---

### 10e. STaR Bootstrapping (Self-Taught Reasoner) - Most Promising

A self-supervised loop: generate forecasts with reasoning chains, filter for the ones that turned out well-calibrated, fine-tune on the successful reasoning chains, repeat. This is particularly appealing because it directly trains on *good reasoning that led to good predictions* - learning the patterns of thought that produce well-calibrated forecasts.

**The cycle:**
1. Forecast a batch of backtest questions, producing reasoning chains + probabilities
2. After resolution, keep only reasoning chains that led to well-calibrated forecasts
3. Fine-tune the model on the successful chains
4. Repeat with the improved model

**Rationalisation:** For questions where the model got it wrong, generate a new reasoning chain *backward* from the correct probability. "Given that the well-calibrated answer was 80%, write a reasoning chain that justifies this." Fine-tune on both naturally successful chains and rationalised chains.

**No human annotation needed.** The only external signal is whether the forecast was well-calibrated, which comes from the resolution data.

**Why this is particularly promising:** Unlike reward-function-based RL (Technique 9), which trains the model to maximise a score, STaR trains the model to *replicate good reasoning*. The model learns: "when I thought about the problem *this way*, I got a well-calibrated answer. When I thought about it *that way*, I didn't." Over iterations, the reasoning patterns that produce good forecasts get reinforced. This is closer to how human forecasters improve - by studying their own successes and failures.

**Evidence:** Zelikman et al. (NeurIPS 2022) showed iterative improvement on reasoning benchmarks without human annotation. Quiet-STaR (2024) extended this with internal reasoning tokens.

**Uncertainty Distillation as a pre-RL step:** Hager et al. (Johns Hopkins, March 2025; arXiv:2503.14749) demonstrated a related approach: sample 100 responses, consolidate semantically equivalent answers, estimate probabilities via Monte Carlo, learn a calibration function, then fine-tune the model to output verbalized confidences. This yields well-calibrated verbalized confidences that outperform baselines 20x slower at inference, and works on black-box models via API fine-tuning. This could serve as a useful bootstrap before the full STaR loop - giving the model a starting point for calibrated probability output before iterating on reasoning quality.

---

## What to Measure

- Does path-independence training (10a) improve calibration without harming accuracy?
- Do counterfactual events (10b) actually improve real-event forecasting?
- Does the poorly-recalled curriculum (10c) avoid contamination while maintaining training signal?
- Does adversarial self-play (10d) improve reasoning quality at inference time?
- Does each STaR iteration (10e) measurably improve calibration on held-out data?
