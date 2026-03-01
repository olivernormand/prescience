# Technique 4: Reasoning Frameworks

**Category:** Inference-time (no model weight changes)
**Prescience stage:** Reasoning (§2.1)

## Overview

Once an agent has gathered information, how should it think about that information to arrive at a probability? The default is just "reason about it and give a number." These are structured alternatives that force the agent to reason in specific ways - decomposing the problem, considering both sides, or building explicit models.

Like search strategies (Technique 3), these are **swappable prompt modules** - different agents in the ensemble can use different reasoning frameworks. The prompting strategy is explicitly experimental (§2.1) and everything here should be tested through the automated experimentation loop.

**Important context:** Schoenegger et al. (2025; arXiv:2506.01578) tested an exhaustive battery of prompt variants - chain-of-thought, base-rate-first, deep-breath + CoT, premortem/promortem analysis, superforecaster persona, conditional odds-ratio, and auto-generated prompts from OpenAI/Anthropic prompt generators. After adjusting for multiple comparisons, **no prompt significantly improved LLM forecasting accuracy**. A superforecaster-authored conditional odds-ratio prompt actually *reduced* accuracy. But approaches that change the *structure* of reasoning (rather than just the wording) can help. The techniques below all change the reasoning structure.

**Where prompting does matter - human-AI collaboration:** Schoenegger et al. (2024) found that human forecasters using a superforecasting LLM assistant improved their accuracy by 23-43% compared to controls. The gains come from structuring human reasoning (Tetlock's "10 commandments of superforecasting"), not from improving LLM output directly. Worth keeping in mind if Prescience ever adds a human-in-the-loop mode.

**The bitter lesson caveat:** As models get better and better, many of these reasoning scaffolds may become unnecessary - the model will internalise good reasoning patterns through scale and training. The real long-term edge for Prescience should come from RL and system architecture, not prompt engineering. These frameworks are worth exploring briefly to see if there's utility, but we shouldn't invest heavily in refining them. If a framework helps, great. If the model already does it naturally, move on.

---

### 4a. Factor Decomposition (PRISM)

Instead of asking "what's the probability of X?", break the question into key factors and estimate each one's contribution separately. Start from the base rate, then adjust for each factor.

**Example:** "Will Country X hold elections by December?"
- Base rate for elections in this region on schedule: 75%
- Factor 1: Constitutional crisis ongoing → -15%
- Factor 2: International pressure for elections → +8%
- Factor 3: Opposition party ready to participate → +5%
- Adjusted estimate: 73%

**Why this might help:** Each factor is essentially its own mini base rate that you can try to estimate from external data. "How often do constitutional crises delay elections?" is a more tractable question than "will this specific election happen?", and you can ground it in historical data. The decomposition makes the reasoning auditable - "is -15% for a constitutional crisis reasonable?" is easier to evaluate than "is 73% overall reasonable?"

**Prompting approach:** Be concrete. Don't ask the agent to "estimate the marginal contribution of each factor" (too abstract). Instead: "For each factor, find a relevant base rate or historical frequency, then estimate how this specific case differs from the base rate, citing evidence."

**Evidence:** PRISM (Probability Reconstruction via Shapley Measures, 2025) outperformed direct prompting across finance, healthcare, agriculture, and sports forecasting. The factor attributions also provide transparency for downstream consumers.

---

### 4b. Argue Both Sides

Before committing to a probability, generate the strongest 3 arguments for YES and the strongest 3 arguments for NO. Only then estimate the probability.

**Why this might help:** Forces the agent to seriously consider the other side before anchoring on a number. Particularly effective with smaller/weaker models that might otherwise latch onto the first piece of evidence they find.

**Caveat:** Frontier models may already do this implicitly. Worth testing whether this adds value with the specific models we're using - it may be that Claude or GPT-4 already considers both sides in their reasoning without being explicitly prompted to. An easy experiment to run.

---

### 4c. Causal Model Extraction

Before estimating a probability, build an explicit causal diagram: what are the key variables, how do they influence each other, and what's the current state of each? Then derive the probability from the model rather than estimating it directly.

**Example:** For "Will inflation fall next quarter?":
- Variable: Central bank policy → currently tightening → pushes inflation down
- Variable: Energy prices → currently rising → pushes inflation up
- Variable: Consumer spending → currently flat → neutral
- Causal links: Bank policy affects spending (lag 2 quarters), energy affects headline but not core
- Derived estimate from propagating through the model

**Why this might help:** Makes the reasoning structure explicit and auditable. Clients can see exactly which variables the system identified and how they connect. Also enables "what if" questions by changing a variable and re-propagating.

**Why this might not help:** Building causal models is hard. The model might construct a plausible-looking diagram that's actually wrong, and then produce a confidently derived probability from a bad model. The diagram adds a layer of apparent rigour that could mask poor judgment.

**Tools:** Probabilistic programming tools like Squiggle can make the causal model executable and inspectable. The AI system can generate a Squiggle program as its forecast representation, which is then executable, auditable, and composable. A Coefficient Giving RFP (November 2025) specifically called for AI tools that produce explicit causal influence diagrams or probabilistic programs - indicating demand for this approach from the forecasting research community.

**Bayesian inversion:** Once a causal model exists, you can apply Bayes' theorem explicitly. Instead of asking "what's the probability of this regulation passing given this committee statement?", decompose it: "If this regulation were going to pass, how likely would this statement have been? And if it weren't going to pass?" The ratio of these likelihoods is the Bayes factor, which tells you exactly how much to update. This is more rigorous than asking the model to "update your prediction" because it forces decomposition into components that can each be independently evaluated.

---

### 4d. Structured Hypothesis Analysis

Enumerate all possible outcomes, then evaluate each piece of evidence for how much it favours each outcome. Focus on **diagnostic evidence** - evidence that distinguishes between outcomes - rather than evidence that's consistent with everything.

This draws from the CIA's Analysis of Competing Hypotheses (ACH) methodology and formal argumentation theory. The key insight: most evidence in a forecasting question is consistent with multiple outcomes and therefore uninformative. The evidence that actually matters is the evidence that strongly favours one outcome over others.

**Example:** For "Will Company X acquire Company Y?"
- Evidence: "CEO mentioned growth strategy" → consistent with acquisition AND organic growth → low diagnostic value
- Evidence: "Regulatory filing for pre-merger notification" → strongly favours acquisition → high diagnostic value
- Focus the analysis on high-diagnostic evidence

**Why this might help:** Prevents the agent from building a long, persuasive narrative from individually weak evidence (a known LLM failure mode). Forces attention onto the evidence that actually discriminates between outcomes.

**Formal argumentation variant:** MQArgEng (2024; arXiv:2405.13036) takes this further - have an LLM generate arguments for/against a claim, transform them into a formal argumentation framework (Dung's framework), compute grounded/preferred extensions using an ASPARTIX solver, and feed surviving arguments back to the LLM. The formal semantics determine which arguments survive scrutiny, grounding the probability in the surviving argument set rather than raw LLM judgment.

**Overlap with other frameworks:** This shares DNA with factor decomposition (4a) and argue-both-sides (4b). The distinctive element is the explicit focus on diagnosticity - not just "what does the evidence say?" but "how much does this evidence distinguish between outcomes?"

---

## Swappable Prompt Modules

All of these frameworks should be implemented as loadable prompt modules that define:
- The reasoning structure (what steps to follow)
- The output format (how to present the reasoning and probability)
- Any domain-specific guidance (which framework works best for which question types)

Different agents in the ensemble can load different modules, providing the reasoning diversity that makes ensembling valuable. The automated experimentation loop tests which modules (and which combinations) produce the best Brier scores, potentially on a per-domain basis.

## What to Measure

- Does any reasoning framework consistently beat unstructured "just reason about it" prompting?
- Do different frameworks work better for different domains?
- Does mixing frameworks across the ensemble improve overall Brier score through diversity?
- Does factor decomposition (4a) produce more auditable and stable forecasts?
