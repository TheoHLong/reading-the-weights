# Research Protocol

This file defines the operating rules for Task D so experiments remain reproducible, interpretable, and publication-oriented.

## Core principle

Do not treat an experiment as evidence unless its setup, artifacts, and interpretation are recoverable from the repository and the local filesystem.

## Must-log items for every run

- Date and local branch
- Commit hash
- Exact command
- Config path
- Device and runtime environment
- Dataset split policy
- Random seed and split seed
- Whether the run is smoke, pilot, ablation, or headline
- Output artifact paths
- Best validation metric
- Final test metric
- Short interpretation: what did we learn and what changed next

## Must-log items for every code change

- Motivation
- Files changed
- Shared interface impact
- Expected effect on experiments
- Verification command

## Required experiment tiers

- `smoke`: fast end-to-end verification, not evidence
- `pilot`: short training used to test directionality
- `ablation`: controlled comparison changing one variable at a time
- `headline`: longer run intended for report-quality conclusions

Do not communicate pilot results to the team as if they are final findings.

## Meaningful progress thresholds

Progress is meaningful enough to communicate when at least one of these is true:

- A shared-interface change is required or has been made.
- A run changes the interpretation of Task D, for example proving the current CIFAR plan is structurally weak.
- A result is replicated across at least two seeds or two comparable runs.
- A new method beats the current best CIFAR baseline by a clear margin and also produces decomposition artifacts.
- A negative result is strong enough to justify redesign, not just more tuning.

## Team communication triggers

Message the team when:

- `build_image_dataloaders` API changed to include validation handling.
- Task D no longer looks like a simple CIFAR extension and needs a revised interpretation.
- We have the first report-quality CIFAR result with both accuracy and decomposition analysis.
- We need to coordinate around shared interfaces or figure-generation expectations.

## Research hygiene rules

- Use validation for model selection and test only for final reporting.
- Keep hardware conditions explicit because infra differences can change outcomes.
- Compare like with like: same split, same seed policy, same epoch budget unless the point is an ablation.
- Prefer small controlled sweeps over one-off heroic runs.
- If a run fails, log the failure mode and whether it is scientific or infrastructural.
- Treat surprising wins as suspect until replicated.

## Publication bar for Task D

Task D starts to look publication-worthy only if we can support one of these claims:

- Bilinear decomposition meaningfully transfers to CIFAR-10 under a clearly specified setup.
- Raw-pixel bilinear models fail on CIFAR-10 for a principled reason, and a redesigned variant fixes the failure.
- Spectral behavior on CIFAR-10 reveals a strong, reproducible contrast with MNIST/Fashion-MNIST and leads to a concrete insight.

Anything weaker is course progress, not paper progress.
