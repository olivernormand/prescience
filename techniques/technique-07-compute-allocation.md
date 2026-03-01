# Technique 7: Compute Allocation

**Category:** Inference-time (system-level)
**Prescience stage:** Pipeline configuration / resource management

## Overview

Not all questions deserve the same amount of compute. Some questions are well-served by a quick base-rate lookup; others need deep multi-agent research. These techniques help decide where to spend resources.

**A caution:** All of these optimisations ultimately risk reducing accuracy. Spending less compute on "easy" questions assumes you can reliably identify which questions are easy before you've actually researched them. Getting that classification wrong means you under-invest in a question that actually needed more work. The baseline of treating all questions equally is simple and safe; these techniques should only be adopted if they demonstrably improve aggregate performance, not just reduce cost.

---

### 7a. Kelly-Weighted Prioritisation

When prediction market prices are available, you can identify where the system's "edge" is highest - questions where you disagree most with the market and have historically been accurate in that domain. These are the questions where additional compute is most valuable, because they represent genuine forecasting skill rather than agreement with consensus.

**How it works:**
1. Produce an initial quick forecast for each question
2. Compare against market prices or consensus
3. Questions with high divergence from consensus (and historical accuracy in that domain) get the full treatment: extended search, multi-agent deliberation, verification
4. Questions where the initial estimate roughly matches consensus get a lighter pass

**For training (RL):** Weight the reward signal by divergence from consensus. A correct prediction that contradicts the market by 30 points is much more informative than one that agrees - it represents genuine forecasting skill.

**Limitation:** Requires available market prices or consensus estimates. Doesn't help for novel questions with no market.

---

### 7b. Fast-and-Frugal Heuristics

Counter-intuitive but empirically supported: for some questions, *less* research is better. Fewer but more diagnostic cues can outperform comprehensive retrieval when evidence is noisy or the question is inherently unpredictable.

**When less is more:**
- Questions in well-established reference classes: the base rate is the best predictor, and additional research adds noise without signal
- Questions with very long time horizons: far-future predictions are dominated by base rates; current news adds mostly noise
- Questions in low-predictability domains: when the environment itself is unpredictable (the Lens Model diagnostic in Technique 6c can identify this), gathering more information doesn't help much

**Application to Prescience:** The question decomposition step (§1.2) could classify questions by expected evidence quality. Questions flagged as "base-rate-dominated" get a frugal treatment (base rate lookup + minimal adjustment). Questions flagged as "evidence-rich" get the full pipeline.

**A "FrugalAgent" variant:** One of the M agents in the ensemble could be configured to use only 2-3 key indicators rather than comprehensive research. If this agent's forecasts are competitive in certain domains, that's a signal to dial back the search intensity there.

The automated experimentation loop (§2.1.1) should test this: for each domain, does the full pipeline actually beat base-rate-only or frugal agents? If not, we're wasting compute.

---

### 7c. Nowcasting Adjustments

Different types of evidence arrive at different speeds. Diplomatic signals appear faster than economic data; social media faster than official reports. When forecasting, the system should account for which evidence streams have likely arrived and which are still pending.

**Example:** If we're forecasting Q3 GDP and it's currently mid-Q3, some leading indicators are available but the actual GDP figure won't be released for months. The system should know this and not treat "no GDP data available" as evidence of anything - it's just evidence that hasn't arrived yet.

**Application to Prescience:** Include awareness of typical evidence timelines in the agent's system prompt. The agent should explicitly note when expected evidence is not yet available rather than treating absence of evidence as evidence of absence.

**Hierarchical borrowing:** When evidence is sparse for a specific question, use information from similar questions (the reference class) to fill in the gaps. This is especially valuable for novel or unusual questions where direct evidence is limited. The decomposition step (§1.2) already identifies reference classes - this uses them more aggressively when question-specific evidence is thin.

---

## What to Measure

- Does Kelly-weighted compute allocation (7a) improve aggregate Brier score compared to uniform allocation?
- For each domain, does the full pipeline beat frugal/base-rate agents (7b)?
- Do nowcasting adjustments (7c) improve forecasts for time-sensitive questions?
