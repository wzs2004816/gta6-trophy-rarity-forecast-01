# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wzs2004816/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wzs2004816/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate at 2026-11-26T00:00:00Z as five quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale. Pure Python 3 standard library, no network, deterministic.

How it works: per trophy, a weighted least-squares fit of logit(earn rate) against log time since launch (weights grow with players tracked and decay with a 24 h half-life, one Huber reweighting pass), averaged 70/30 with a linear-in-time fit to catch the convex early part of an adoption curve; precision-weighted blending of that extrapolation with a difficulty prior built from the game's structure (91 story missions over 5 chapters, a 50–80 h main story, ~200 h for 100%, GTA V's long-run rarity funnel); the story funnel (chapter 1 ≥ chapter 2 ≥ … ≥ final mission ≥ 100% ≥ platinum) enforced on the medians; quantiles from the honest predictive spread, which widens with the horizon.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=zy5dg92r&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
