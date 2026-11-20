# GTA VI trophy rarity forecast — v3

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bnbf/gta6-trophy-rarity-forecast-01-bot-v3/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bnbf/gta6-trophy-rarity-forecast-01-bot-v3/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for make.fun's **gta6-trophy-rarity-forecast-01** bounty: forecast every
GTA VI PlayStation trophy's Exophase earn rate at 2026-11-26T00:00:00Z, as five
quantiles per trophy, scored by mean pinball loss on the logit scale.

## Method (`main.py`, Python 3.13 stdlib only)

Exophase's page only changes when it refreshes (about every 4 hours), so a
case's history is a step function: many reads, few distinct rate vectors.
Each trophy's distinct vectors, at their first-seen times, are the
observations; the target is the rate at the case's target instant.

- **Trending trophies** (the history moves): two curve families are fitted
  to the observations, and a short-range extrapolation test picks one.
  - Saturating exponential `r(t) = rinf - (rinf - r0) * exp(-t/tau)`,
    `r0` fixed to the first observed rate, `tau` by unweighted squared
    error, `rinf` by player-count-weighted least squares.
  - Power law `r(t) = r0 * (1 + t/tau) ** gamma`, `(tau, gamma)` by
    player-count-weighted squared error over a grid.
  - The pick is leave-last-4-out cross-validation in logit space: each
    family is refit on the earlier points and scored on the last four;
    the lower error wins. In-sample fit alone cannot tell the families
    apart (both fit the window; they only diverge past it), so the
    hold-out test is what separates them. When too few states exist to
    cross-validate, the two families are averaged on the logit scale.
  - Quantiles are the picked median on the logit scale with a spread of
    `0.06 + 0.15 * sqrt(CV error)` — trophies the families cannot pin
    down get wider quantiles.
- **Frozen pages** (every read identical, as in the practice cases): the
  page has not refreshed since launch night, so the target sits just
  after a refresh that dilutes every rate. The refresh takes tracked
  players from 638 to about 3,868 while the newly tracked players have
  earned almost nothing, so the forecast is
  `(638 * r + 3230 * 0.05 * r**4) / 3868`, with a 0.15 logit spread.

Deterministic, no network, no randomness. Runs in ~0.2 s per case
(limit 8 s); outputs are byte-identical across runs.

## Verification (local, `~/workspace/makefun/fx/score.py`)

- Practice: **p00 0.0780, p01 0.0780** (v2: 0.0946 on both).
- Held-out synthetic final-like cases (5 shapes, never used for tuning):
  **f00 0.0289, f00_b 0.0288, f00_c 0.0315, f00_d 0.0709, f00_e 0.0554 —
  mean 0.0431, worst 0.0709** (v2: mean 0.0934, worst 0.1053).
- Extra tuning synthetic cases (6 fresh seeds): mean 0.0453, worst 0.0602.
- Determinism: two runs byte-identical on all cases. Output sizes
  ~2.8 KB (limit 65,536 B).

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes one `<id>.out` per case.
