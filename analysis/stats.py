"""Statistics for factorial experiments with one observation per run.

descriptive   mean, SD, 95 % CI of the mean, P95, max per treatment
anova         factorial ANOVA (Type III, sum-to-zero contrasts) with partial eta squared
assumptions   Shapiro-Wilk on the residuals, Levene (Brown-Forsythe) across treatments
kruskal       Kruskal-Wallis across treatments (non-parametric fallback)
tukey         Tukey HSD for a significant main effect or interaction
logistic      binomial GLM (quasi-binomial scale) on counts out of a total
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as st
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ALPHA = 0.05


def _term(factor):
    return f"C({factor}, Sum)"


def formula(response, factors, interactions=True):
    terms = [_term(f) for f in factors]
    return f"{response} ~ " + (" * ".join(terms) if interactions else " + ".join(terms))


def descriptive(df, factors, response):
    rows = []
    for levels, g in df.groupby(factors, sort=True, observed=True):
        y = g[response].dropna().to_numpy(dtype=float)
        levels = levels if isinstance(levels, tuple) else (levels,)
        n = len(y)
        mean = y.mean() if n else np.nan
        sd = y.std(ddof=1) if n > 1 else np.nan
        half = st.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        rows.append({**dict(zip(factors, levels)), "n": n, "mean": mean, "sd": sd,
                     "ci95_low": mean - half, "ci95_high": mean + half,
                     "p95": np.percentile(y, 95) if n else np.nan, "max": y.max() if n else np.nan})
    return pd.DataFrame(rows)


def replicates(df, factors):
    return int(df.groupby(factors, observed=True).size().min())


def anova(df, factors, response, interactions=None):
    """(ANOVA table with F, p and partial eta squared, fitted model).

    Interactions are included when every treatment has at least 2 runs; with one
    run per treatment (experiment C) only main effects can be estimated.
    """
    data = df.dropna(subset=[response])
    if interactions is None:
        interactions = replicates(data, factors) >= 2
    model = smf.ols(formula(response, factors, interactions), data=data).fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = sm.stats.anova_lm(model, typ=3)
    table = table.drop(index="Intercept", errors="ignore")
    ss_res = table.loc["Residual", "sum_sq"]
    table["partial_eta2"] = table["sum_sq"] / (table["sum_sq"] + ss_res)
    table.loc["Residual", "partial_eta2"] = np.nan
    table.index = [_clean(i) for i in table.index]
    table = table.rename(columns={"PR(>F)": "p"})
    table["significant"] = table["p"] < ALPHA
    return table, model


def _clean(term):
    return term.replace(", Sum)", "").replace("C(", "").replace(":", " x ")


def assumptions(model, df, factors, response):
    data = df.dropna(subset=[response])
    resid = model.resid
    shapiro_p = st.shapiro(resid).pvalue if len(resid) >= 3 and np.ptp(resid) > 0 else np.nan
    groups = [g[response].to_numpy(dtype=float) for _, g in data.groupby(factors, observed=True) if len(g) >= 2]
    levene_p = np.nan
    if len(groups) >= 2 and any(np.ptp(g) > 0 for g in groups):
        levene_p = st.levene(*groups, center="median").pvalue
    ok = not (shapiro_p < ALPHA) and not (levene_p < ALPHA)
    return {"shapiro_p": shapiro_p, "levene_p": levene_p, "ok": bool(ok)}


def kruskal(df, factors, response):
    groups = [g[response].dropna().to_numpy(dtype=float) for _, g in df.groupby(factors, observed=True)]
    groups = [g for g in groups if len(g)]
    if len(groups) < 2 or np.ptp(np.concatenate(groups)) == 0:
        return {"H": np.nan, "p": np.nan}
    r = st.kruskal(*groups)
    return {"H": float(r.statistic), "p": float(r.pvalue)}


def tukey(df, term_factors, response):
    """Pairwise Tukey HSD between the levels (or level combinations) of term_factors."""
    data = df.dropna(subset=[response])
    groups = data[list(term_factors)].astype(str).agg(" / ".join, axis=1)
    with warnings.catch_warnings():  # groups with zero variance (deterministic responses)
        warnings.simplefilter("ignore")
        res = pairwise_tukeyhsd(data[response].to_numpy(dtype=float), groups.to_numpy(), alpha=ALPHA)
    head, *rows = res._results_table.data
    out = pd.DataFrame(rows, columns=head)
    return out.rename(columns={"meandiff": "mean_diff", "p-adj": "p_adj", "reject": "significant"})


def significant_terms(table, factors):
    """Factor tuples of the significant model terms, highest order first."""
    terms = []
    for name, row in table.drop(index="Residual").iterrows():
        if row["significant"]:
            terms.append(tuple(name.split(" x ")))
    return sorted(terms, key=len, reverse=True)


def logistic(df, factors, successes, total, interactions=None):
    """Quasi-binomial GLM of successes/total; Wald chi-square test per term.

    Used for proportions (audio blocks lost, keywords missed) when ANOVA on
    percentages is not appropriate. Returns None when the data have no variation.
    """
    data = df.dropna(subset=[successes, total]).copy()
    data = data[data[total] > 0]
    data["_prop"] = data[successes] / data[total]
    if data["_prop"].nunique() < 2:
        return None
    if interactions is None:
        interactions = replicates(data, factors) >= 2
    model = smf.glm(formula("_prop", factors, interactions), data=data,
                    family=sm.families.Binomial(), var_weights=data[total])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(scale="X2")
            wald = res.wald_test_terms(scalar=True).summary_frame()
    except (np.linalg.LinAlgError, ValueError):
        return None
    wald = wald.drop(index="Intercept", errors="ignore")
    wald.index = [_clean(i) for i in wald.index]
    wald = wald.rename(columns={"statistic": "chi2", "P>chi2": "p", "df_constraint": "df", "df constraint": "df"})
    wald["significant"] = wald["p"] < ALPHA
    return wald[["chi2", "df", "p", "significant"]]
