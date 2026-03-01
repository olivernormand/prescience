# Technique 4: Reasoning Frameworks

**Category:** Inference-time (no model weight changes)
**Prescience stage:** Reasoning (§2.1)

## Overview

**Tetlock Commandments:** 2 (break into sub-problems → 4a), 3 (inside/outside view → 4a), 4 (balance evidence reaction → 4e), 5 (clashing causal forces → 4b, 4d)

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

**Making it executable:** Tools like Squiggle can turn the causal model into a runnable program rather than just a diagram. The forecast becomes code: "inflation = f(central_bank_policy, energy_prices, consumer_spending)" with distributions on each input. This is auditable (clients can inspect and challenge each assumption), composable (combine models across related questions), and updatable (change one variable and see how the forecast shifts). A Coefficient Giving RFP (November 2025) specifically called for AI tools that produce explicit causal influence diagrams - indicating demand for this from institutional forecast consumers.

**Bayesian updating within the model:** Once a causal model exists, you can update it rigorously as new evidence arrives. Instead of asking the model to vaguely "update your prediction given this news", decompose it: "If inflation were going to fall, how likely would this jobs report have been? And if inflation were going to stay high, how likely?" The ratio of these likelihoods tells you exactly how much to shift the probability. This is more rigorous than re-running the whole pipeline - it forces the agent to think about what the evidence means *relative to each possible outcome*, not just whether it "seems bullish or bearish."

---

### 4d. Structured Hypothesis Analysis

Enumerate all possible outcomes, then evaluate each piece of evidence for how much it favours each outcome. Focus on **diagnostic evidence** - evidence that distinguishes between outcomes - rather than evidence that's consistent with everything.

This draws from the CIA's Analysis of Competing Hypotheses (ACH) methodology and formal argumentation theory. The key insight: most evidence in a forecasting question is consistent with multiple outcomes and therefore uninformative. The evidence that actually matters is the evidence that strongly favours one outcome over others.

**Example:** For "Will Company X acquire Company Y?"
- Evidence: "CEO mentioned growth strategy" → consistent with acquisition AND organic growth → low diagnostic value
- Evidence: "Regulatory filing for pre-merger notification" → strongly favours acquisition → high diagnostic value
- Focus the analysis on high-diagnostic evidence

**Why this might help:** Prevents the agent from building a long, persuasive narrative from individually weak evidence (a known LLM failure mode). Forces attention onto the evidence that actually discriminates between outcomes.

**Formal stress-testing variant:** MQArgEng (2024; arXiv:2405.13036) takes this further - generate arguments for and against, then formally check which arguments "survive" when they attack each other. If the argument for YES depends on an assumption that the argument for NO undermines, the YES argument doesn't survive. The probability is then grounded in whatever arguments remain standing, not in the full set of arguments the model initially generated. This is heavier machinery than the other frameworks, but the core idea - test whether arguments actually survive scrutiny rather than just listing them - is sound.

**Overlap with other frameworks:** This shares DNA with factor decomposition (4a) and argue-both-sides (4b). The distinctive element is the explicit focus on diagnosticity - not just "what does the evidence say?" but "how much does this evidence distinguish between outcomes?"

---

### 4e. Explicit Bayesian Updating

Instead of asking the agent to holistically estimate a probability, force it to reason in terms of priors and likelihood ratios using Bayes' theorem:

**P(outcome | evidence) = P(evidence | outcome) × P(outcome) / P(evidence)**

For each significant piece of evidence E, the agent must answer three questions:
1. What's my prior probability for this outcome? P(outcome)
2. If the outcome were going to happen, how likely would I be to see this evidence? P(E | outcome)
3. If the outcome were NOT going to happen, how likely would I be to see this evidence? P(E | ¬outcome)

The posterior is then computed mechanically from the likelihood ratio: P(E | outcome) / P(E | ¬outcome). Evidence with a ratio near 1 is uninformative regardless of how relevant it *seems*. Evidence with a ratio far from 1 is diagnostic and should move the probability substantially.

**Why this might help:** Models systematically under-update relative to Bayes' theorem. The Bayesian Coherence Coefficient (arXiv:2507.17951, 2025) found r=0.906 correlation between model size and Bayesian coherence - larger models are closer but still fall short. Explicitly prompting the likelihood ratio decomposition forces the agent to think about what the evidence means *relative to each possible outcome*, rather than just asking "does this evidence seem bullish or bearish?" It also naturally surfaces the distinction between diagnostic and non-diagnostic evidence (see Technique 4d).

**Connection to path independence:** Bayesian updating is mathematically commutative - updating on evidence A then B produces the same posterior as updating on B then A. If agents reason in explicitly Bayesian terms, path independence (Technique 10a) comes for free. This is the theoretical justification for why path-dependent forecasting is irrational (De Finetti's Dutch book theorem).

**Practical prompting:** The prompt should walk the agent through each update step by step. For a question like "Will Country X default on its debt by December?":
- Prior: base rate of sovereign defaults in this credit rating band → 8%
- Evidence 1: IMF just downgraded growth forecast. P(downgrade | default) ≈ 0.6, P(downgrade | no default) ≈ 0.15. Likelihood ratio = 4. Posterior → ~26%
- Evidence 2: Government just secured emergency bilateral loan. P(loan | default) ≈ 0.3, P(loan | no default) ≈ 0.5. Likelihood ratio = 0.6. Posterior → ~18%

Each step is auditable - someone can challenge the individual likelihood estimates without having to argue about the holistic probability.

**When this might not help:** Estimating likelihood ratios P(E|H) is itself hard, and the agent may just shift its uncertainty from "what's the probability?" to "what's the likelihood ratio?" If the agent can't produce reasonable likelihood estimates, the formal structure doesn't buy much. Also, complex questions with many interacting pieces of evidence may not decompose cleanly into sequential Bayesian updates - the evidence may be correlated in ways that make naive sequential updating incorrect.

**Relationship to other frameworks:** This can be combined with factor decomposition (4a) - use factor decomposition to identify the key evidence, then use Bayesian updating to process each piece. It's also complementary to structured hypothesis analysis (4d) - the likelihood ratio naturally measures diagnosticity.

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
- Does explicit Bayesian updating (4e) reduce the under-updating problem identified by the Bayesian Coherence Coefficient?
- Do agents using Bayesian updating exhibit greater path independence than those using holistic estimation?
