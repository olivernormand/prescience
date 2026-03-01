# Technique 6: Error Diagnostics

**Category:** Diagnostic (no model weight changes)
**Prescience stage:** Evaluation (§3.5) / automated experimentation loop

## Overview

**Tetlock Commandments:** 8 (look for errors behind your mistakes), 10 (master the error-balancing bicycle through deliberative practice)

When the system gets a forecast wrong, understanding *why* it was wrong determines what to fix. These diagnostic tools decompose forecast errors into distinct components, each pointing to a different intervention. Without them, you're guessing at what to improve.

---

### 6a. BIN Decomposition (Bias, Information, Noise)

Break forecast errors into three components:

- **Bias:** Systematic directional errors. The system consistently predicts 60% for events that happen 80% of the time. This is a calibration problem - relatively cheap to fix with Platt scaling or per-domain calibration (Technique 1)
- **Information:** How much signal the system captures about the truth. Can the system distinguish between events that are likely and events that are unlikely, or does it predict ~50% for everything? Low information means the system needs better search, better reasoning, or better data sources - the expensive fixes
- **Noise:** Random variation. If you ran the system twice on the same question, how different would the answers be? High noise means the system is unstable - more ensembling or consistency constraints would help

**The formula:** Brier score = Bias² + Noise - Information

**Why this matters for Prescience:** Each component points to a different intervention. The automated experimentation loop (§2.1.1) should run BIN decomposition after every backtest and use it to decide what to work on next:
- High bias? → Improve calibration (cheap)
- Low information? → Improve search and reasoning (expensive but high-value)
- High noise? → Increase ensemble size or add consistency checks (moderate cost)

**Practical difficulty:** BIN decomposition requires a reasonable number of resolved forecasts to be statistically meaningful - you need enough data points in each segment to trust the decomposition. With small evaluation sets, the decomposition itself will be noisy. Start with the aggregate decomposition and only segment by domain once we have enough data per domain (probably 50+ resolved questions each).

**Segment the analysis:** Once you have enough data, don't just look at aggregate BIN. Break it down by domain (geopolitics, economics, tech), time horizon (weeks, months, quarters), and question difficulty. The dominant error source often varies across segments - the system might be well-calibrated on politics but poorly informed on economics.

---

### 6b. Brier Score Decomposition (Reliability, Resolution, Uncertainty)

A related decomposition from meteorology (Murphy, 1973):

**Brier score = Reliability - Resolution + Uncertainty**

- **Reliability:** How well-calibrated are the forecasts? When the system says 70%, does the event happen 70% of the time? Poor reliability → calibration fixes
- **Resolution:** How much do the forecasts vary? A system that always predicts 50% has zero resolution. A system that confidently says 90% or 10% when appropriate has high resolution. Poor resolution → the system isn't discriminating between likely and unlikely events, needs better information
- **Uncertainty:** How inherently unpredictable are the questions? This is a property of the question set, not the system. It sets the floor on achievable Brier score

**When to use which:** BIN (6a) is better for diagnosing what to fix next. Brier decomposition (6b) is better for understanding whether the problem is calibration or discrimination, and for benchmarking against the theoretical floor (uncertainty) - i.e. how much of our error is because the questions are inherently unpredictable vs because we're forecasting badly. Stephenson et al. (2008) identified additional within-bin components that make this decomposition more robust for smaller samples.

**Real-time quality checks (no resolved outcomes needed):** The Martingale Score and Bayesian Coherence Coefficient (discussed in Technique 5c) can also feed into diagnostic analysis - if the system consistently fails to update on evidence in a specific domain, that shows up as an information problem in BIN decomposition when outcomes eventually resolve. Tracking both the real-time signals and the post-resolution decomposition lets you validate whether your pre-resolution quality flags actually predict post-resolution errors.

---

### 6c. The Lens Model

A more fine-grained diagnostic that asks: is the system using the right cues, or is it using the right cues inconsistently?

- **Ecological validity:** How well do the cues (evidence) the system uses actually predict outcomes? If the system relies on evidence that isn't actually predictive, no amount of reasoning improvement will help - it needs different data sources
- **Cue utilisation:** How consistently does the system use the cues it has? If the system sometimes weighs a factor heavily and sometimes ignores it for similar questions, that's inconsistency
- **Matching:** How well does the system's cue usage align with what the cues actually predict?

**Practical difficulty:** The hard part is defining what the "cues" are. In weather forecasting, the cues are clearly defined variables (temperature, pressure, humidity). In event forecasting, it's much less clear what counts as a cue. You'd need to extract structured features from the reasoning chains (e.g. "did the agent cite base rates?", "did it reference prediction markets?", "how many sources did it use?") and then evaluate which of those features actually predict outcomes. This is doable but requires careful feature engineering.

**When to use:** When you've narrowed down a domain with poor performance and want to understand whether the system is looking at the wrong things (need different evidence), using the right things inconsistently (need better reasoning), or facing a genuinely unpredictable domain (accept the limit). This is a deeper investigation tool, not something to run routinely.

---

## What to Measure

Run all diagnostics after every backtest evaluation. Track how the components change as we iterate on the system. The progression should be:

1. BIN decomposition identifies the dominant error source
2. We intervene (calibration, search improvement, or ensembling)
3. Next backtest shows whether the targeted component improved
4. Repeat on the next-largest component

This is the engine of the automated experimentation loop.
