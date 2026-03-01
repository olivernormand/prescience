# Technique 8: Walk-Forward Optimisation

**Category:** System-level
**Prescience stage:** Automated experimentation loop (§2.1.1, §7.1)

## Overview

The automated improvement loop, made concrete. Forecast one period's events, analyse what went wrong, generate improvements, apply them to the next period, repeat. This is how the system gets better over time - replicating the experience loop that makes human superforecasters good.

## How It Works

1. **Forecast:** System forecasts February events using only information available in January
2. **Evaluate:** After February events resolve, run BIN decomposition (Technique 6a). What was the dominant error source? Which domains performed worst?
3. **Diagnose:** Was it a calibration problem (bias)? An information problem (bad search)? A noise problem (inconsistent agents)?
4. **Hypothesise:** Generate a specific, testable improvement. "Adding dual-perspective search for geopolitical questions should reduce the information gap in that domain"
5. **Test:** Forecast March events with the improvement applied. Compare against the March baseline
6. **Accept/reject:** If the improvement helps, keep it. If not, revert
7. **Repeat:** Move to the next hypothesis

## Application to Prescience

This is the core mechanism described in §2.1.1 of the README. The key design requirements:

**It should be trivial to run.** A single command should backtest the system against a standard question set and produce a Brier score. If this is hard to do, experimentation won't happen.

**Everything tunable should be configurable.** Prompts, agent count, aggregation method, calibration parameters, data source configuration, reasoning frameworks - all should be changeable via config and measurable through the backtest loop.

**Full provenance logging.** Every experiment should record: what changed, what the hypothesis was, what the result was, and the full config that produced it. This makes it possible to trace what worked and why.

**Automated vs. manual:** Start with manual failure analysis (human reviews backtest results and identifies improvements). Once we understand what kinds of improvements actually generalise, encode those patterns into an automated loop that an agent (e.g. Claude Code) can run independently.

## Guard-Rails Against Overfitting

- **Never test and train on the same period.** This is the whole point of walk-forward - each improvement is tested on unseen data
- **Track forward generalisation.** Do improvements on February data actually help on March data? Or are they period-specific flukes?
- **Maintain a held-out test set** that's never used for optimisation
- **Be suspicious of small improvements.** With noisy data, a 0.5% Brier improvement could easily be chance. Set a significance threshold

## What to Measure

- Is system Brier score improving over successive walk-forward periods?
- Are improvements generalising forward (do gains on period T persist on period T+1)?
- Which types of changes have the highest success rate?
- How many experiments does it take to find a meaningful improvement?
