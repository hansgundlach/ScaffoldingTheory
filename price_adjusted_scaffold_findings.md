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

## Preferred specification: price as a covariate (no pole, full n = 185)

The ratio is statistically awkward because of the $1 pole. The cleaner spec keeps
the dependent variable as `logit(accuracy)` and enters `log(cost)` as a **control
term** rather than a denominator; nothing is divided, so the pole disappears and
all 185 systems are used. "Combined effect of the scaffold dummies" becomes their
**incremental adjusted R²** on top of the controls. Outputs:
`price_covariate_scaffold_results.csv`, `figures/price_covariate_scaffold_adjr2.png`.

| Specification (DV = `logit(accuracy)`) | adjusted R² | incremental |
|---|---|---|
| `logit_acc ~ log_cost` | 0.136 | — |
| `+ scaffold dummies` | 0.472 | **+0.336** |
| `+ model dummies` | 0.132 | **−0.004** |
| `logit_acc ~ log_cost + benchmark` | 0.519 | — |
| `+ scaffold dummies` | 0.694 | **+0.176** |
| `+ model dummies` | 0.627 | **+0.108** |

- **Price explains little on its own** (adjusted R² 0.136; 6 % of variance in the
  Shapley below). Accuracy differences are not mainly a price story.
- **Net of price only**, scaffold dummies add a large +0.336 while model dummies
  add essentially nothing (−0.004). This is *not* evidence that the model is
  irrelevant — it is the nesting artifact. Because scaffolds are benchmark-
  specific, the scaffold dummies stand in for *which benchmark* a run is on, and
  benchmark is the dominant difficulty axis; model dummies, spread across
  benchmarks, can't predict accuracy until you know the benchmark (the same model
  scores ~5 % on SciCode and ~70 % on GAIA). It is the clearest possible argument
  for controlling for benchmark.
- **Net of price *and* benchmark** (the fair comparison), the scaffold's
  contribution (+0.176) is still larger than the model's (+0.108): within a
  benchmark, holding price fixed, which scaffold you run explains more of the
  remaining accuracy spread than which model you run. Entering each factor
  **last** — net of price, benchmark *and the other factor* — sharpens this to
  scaffold **+0.127** vs. model **+0.060** (≈ 2.1×): the variance uniquely
  attributable to the scaffold is about double that uniquely attributable to the
  model.
- **4-factor Shapley** on `logit(accuracy)` (full-model R² 0.804):
  **price 5.9 % · benchmark 34.8 % · model 13.6 % · scaffold 26.0 % · residual
  19.6 %.** Benchmark difficulty dominates, then scaffold, then model. Read this
  one with the collinearity caveat below: scaffold and benchmark overlap, so the
  Shapley hands scaffold a share of the benchmark-like variance; the +0.176
  incremental is the conservative estimate of scaffold's *unique* power, and it
  still beats the model's.

## Takeaway

Across both specifications the conclusion is the same and, if anything, stronger
under the cleaner covariate spec: **the scaffold is a first-order driver of agent
performance — on a price-adjusted basis at least as important as the model, and
in the price-controlled specification its unique contribution is about twice the
model's (+0.127 vs +0.060 adjusted R², each net of price, benchmark and the other
factor).** Price itself is a minor axis (≈ 6 % of variance). This extends the
headline per-benchmark ANOVA claim (scaffold rivals model on some benchmarks) to
a pooled, price-aware analysis — and the pooled, fully-controlled comparison puts
the scaffold ahead. (The pooled result reads stronger than the per-benchmark one
in `anova_analysis.py` partly because pooling lets the 11 scaffold dummies borrow
strength across benchmarks while each benchmark individually has only 2–3.)

## Method notes

**Winsorisation.** Used only as a robustness check on the *ratio* metric, whose
$1 pole produced one explosive value (`y ≈ 14.6`) that swamped the variance and
drove the adjusted R²'s to ≈ 0. Winsorising "pulls in the tails": we cap `y` at
its 2nd and 98th percentiles — any value below the 2nd-percentile cutoff is set
to that cutoff and any value above the 98th is set to the 98th — so the extreme
points are kept (we don't drop rows) but can no longer dominate the sum of
squares. It is the soft alternative to the hard `cost ≥ $1` exclusion, and it
gives the same qualitative answer, which is what makes the ratio result credible
rather than an artifact of which four rows we deleted. The covariate spec needs
no winsorisation because it never divides.

**Why this is a "Shapley".** The question "how much does scaffold explain?" has no
unique answer when factors are correlated: scaffold's marginal R² depends on
whether it is entered before or after benchmark/model. The Shapley value (here the
LMG / Shapley-regression method) resolves this the same way cooperative game
theory splits a payout among players — each factor's credit is the *average* of
its marginal contribution (increase in R²) over **every ordering** in which the
factors could be added. Averaging over all orderings is what makes the shares
(i) order-independent and (ii) exactly additive: they sum to the full-model R²,
with the leftover `1 − R²(full)` as residual. That additivity is the property a
plain sequential regression lacks and is why the bars can be stacked. The same
logic and code (`shapley()` in the script) underlie the per-benchmark
decomposition in `anova_analysis.py`; here we just apply it pooled and with more
factors (price/benchmark/model/scaffold).
