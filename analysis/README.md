# analysis

Analysis of `experiments/results/` with SciPy and statsmodels:

1. Descriptive statistics per treatment: mean, standard deviation, 95 % CI, P95 and maximum; box plots.
2. Factorial ANOVA with interactions and effect size (partial η²).
3. Assumption checks: Shapiro-Wilk and Levene; if they fail, log transform or a non-parametric alternative.
4. Multiple comparisons with Tukey HSD.
5. Proportions (audio loss, missed words) with logistic regression when ANOVA does not apply.

Each observation is the summary value of one run, not an individual inference.
