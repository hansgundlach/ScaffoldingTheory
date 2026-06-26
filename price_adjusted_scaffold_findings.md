# Price-adjusted variance decomposition: scaffold vs. model

Companion to the Shapley/ANOVA decomposition in `anova_analysis.py`. That
analysis decomposes `logit(accuracy)`; this one folds price into a single
efficiency metric and asks the same model-vs-scaffold question.

**Script:** `price_adjusted_scaffold_regression.py`
**Outputs:** `price_adjusted_scaffold_results.csv`, `figures/price_adjusted_scaffold_adjr2.png`

## What was run

Pooling all systems across the nine HAL benchmarks (one row per
benchmark × scaffold × model, keeping the best-accuracy run of any High/non-High
duplicate, the same collapse `anova_analysis.py` uses), we build a
price-adjusted performance metric

```
y = logit(accuracy) / log(cost)        ("logit performance per log dollar")
```

and regress it on a full set of **scaffold dummies** (one per scaffold).
The headline number is the regression's **adjusted R²** — the combined
explanatory power of all scaffold dummies for `y`. We report the same number
for **model dummies**. Adjusted R² is the right yardstick for the comparison
because it already penalises the model factor for having more levels (19 models →
18 dummies) than the scaffold factor (12 scaffolds → 11 dummies), so the two are
compared on equal footing despite the different dummy counts.

Two things complicate a naive reading and are handled explicitly:

- **Pole at $1.** `log(cost)` is zero at cost = $1 and negative below it, so `y`
  explodes for cheap, low-accuracy runs and flips sign for the four sub-$1 runs.
  One point (SciCode / DeepSeek V3, 3.08 % at $0.79) gives `y ≈ 14.6` while every
  other system lies in roughly [−4.6, +2]. On the full sample that single point
  inflates the total variance enough to push both adjusted R²'s to ≈ 0, so the
  **primary specification drops cost < $1** (n = 181). A winsorised variant (clip
  `y` to its 2nd/98th percentile, n = 185) gives the same qualitative answer.
- **Nesting.** Most scaffolds are benchmark-specific (CORE-Agent only on
  CORE-bench, the SciCode agents only on SciCode, …) while models recur across
  benchmarks. A raw `y ~ scaffold` therefore lets scaffold dummies absorb
  benchmark-difficulty variance. We separate this with (a) incremental adjusted
  R² after benchmark fixed effects and (b) a 3-factor Shapley decomposition
  (benchmark / model / scaffold).

## Results

Primary specification (cost ≥ $1, n = 181):

| Regressor (DV = `logit(acc)/log(cost)`) | R² | **adjusted R²** |
|---|---|---|
| **Scaffold dummies** | 0.310 | **0.265** |
| **Model dummies** | 0.331 | **0.257** |

- **Scaffolds alone explain about as much of price-adjusted performance as
  models do** — adjusted R² 0.265 (scaffold) vs. 0.257 (model). On this
  price-aware metric the two factors are essentially tied.
- **Once benchmark is controlled for**, the model carries more *incremental*
  weight than the scaffold (incremental adjusted R² +0.318 for model vs. +0.124
  for scaffold): part of the raw scaffold R² is benchmark-difficulty leaking
  through the benchmark-specific scaffolds.
- The **3-factor Shapley** decomposition of the full-model R² (0.729) splits as
  **benchmark 15.9 % · model 33.9 % · scaffold 23.1 % · residual 27.1 %** — the
  model is the larger single driver, but the scaffold accounts for roughly
  two-thirds as much explained variance (0.68×).
- **Robustness.** The full sample (n = 185) is dominated by the cost ≈ $1 outlier
  and is uninformative (both adjusted R² ≈ −0.02). Winsorising instead of
  dropping reproduces the picture: scaffold 0.158, model 0.168 (raw adjusted R²),
  Shapley model 23.2 % vs. scaffold 13.5 %.

## Takeaway

On a price-adjusted metric, **the scaffold is a first-order driver of
cost-efficient agent performance — about as important as the model in raw terms,
and roughly two-thirds as important once benchmark difficulty is partialled out.**
This extends the headline ANOVA claim (scaffold rivals model on some benchmarks)
to a metric that explicitly accounts for cost.

> Caveat for the writeup: the `logit / log(cost)` ratio is statistically awkward
> because of the $1 pole; the numbers above use the cost ≥ $1 sample. If a
> cleaner specification is wanted, regressing `logit(accuracy)` on scaffold/model
> dummies **with `log(cost)` as a covariate** (price as a control rather than a
> denominator) avoids the pole entirely and answers the same question.
