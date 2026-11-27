import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stats as S  # noqa: E402


def factorial(effect_a=5.0, effect_b=0.0, interaction=0.0, reps=5, sd=1.0, seed=0):
    """y = 10 + effect_a*[a=a2] + effect_b*b + interaction*[a=a2]*b + noise, a in 2 levels, b in 3."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in ("a1", "a2"):
        for i, b in enumerate(("b1", "b2", "b3")):
            for _ in range(reps):
                y = 10 + effect_a * (a == "a2") + effect_b * i + interaction * (a == "a2") * i + rng.normal(0, sd)
                rows.append({"A": a, "B": b, "y": y})
    return pd.DataFrame(rows)


class TestStats(unittest.TestCase):
    def test_descriptive(self):
        df = factorial(reps=4)
        d = S.descriptive(df, ["A", "B"], "y")
        self.assertEqual(len(d), 6)
        y = df[(df.A == "a1") & (df.B == "b1")]["y"].to_numpy()
        row = d.iloc[0]
        self.assertAlmostEqual(row["mean"], y.mean())
        self.assertAlmostEqual(row["sd"], y.std(ddof=1))
        half = st.t.ppf(0.975, 3) * y.std(ddof=1) / 2
        self.assertAlmostEqual(row["ci95_high"] - row["mean"], half)

    def test_anova_finds_the_real_effect_only(self):
        table, _ = S.anova(factorial(effect_a=5, effect_b=0), ["A", "B"], "y")
        self.assertTrue(table.loc["A", "significant"])
        self.assertFalse(table.loc["B", "significant"])
        self.assertFalse(table.loc["A x B", "significant"])
        self.assertEqual(table.loc["Residual", "df"], 24)
        self.assertTrue(((table["partial_eta2"].dropna() >= 0) & (table["partial_eta2"].dropna() <= 1)).all())

    def test_anova_detects_interaction(self):
        table, _ = S.anova(factorial(effect_a=0, interaction=4), ["A", "B"], "y")
        self.assertTrue(table.loc["A x B", "significant"])
        self.assertEqual(S.significant_terms(table, ["A", "B"])[0], ("A", "B"))

    def test_partial_eta2_matches_one_way_formula(self):
        df = factorial(effect_a=3, reps=6)
        table, _ = S.anova(df, ["A"], "y")
        grand = df.y.mean()
        ss_a = sum(len(g) * (g.y.mean() - grand) ** 2 for _, g in df.groupby("A"))
        ss_res = sum(((g.y - g.y.mean()) ** 2).sum() for _, g in df.groupby("A"))
        self.assertAlmostEqual(table.loc["A", "partial_eta2"], ss_a / (ss_a + ss_res))

    def test_one_run_per_treatment_uses_main_effects(self):
        df = factorial(reps=1)
        table, _ = S.anova(df, ["A", "B"], "y")
        self.assertNotIn("A x B", table.index)
        self.assertEqual(S.replicates(df, ["A", "B"]), 1)

    def test_assumptions(self):
        df = factorial()
        _, model = S.anova(df, ["A", "B"], "y")
        self.assertTrue(S.assumptions(model, df, ["A", "B"], "y")["ok"])
        skewed = df.assign(y=np.exp(df.y))  # strongly non-normal, unequal variances
        _, model = S.anova(skewed, ["A", "B"], "y")
        self.assertFalse(S.assumptions(model, skewed, ["A", "B"], "y")["ok"])

    def test_kruskal(self):
        self.assertLess(S.kruskal(factorial(effect_a=5), ["A", "B"], "y")["p"], 0.05)

    def test_tukey(self):
        tk = S.tukey(factorial(effect_a=5), ("A",), "y")
        self.assertEqual(len(tk), 1)
        self.assertTrue(bool(tk["significant"].iloc[0]))
        self.assertAlmostEqual(tk["mean_diff"].iloc[0], 5, delta=1)

    def test_logistic(self):
        rng = np.random.default_rng(1)
        rows = [{"A": a, "B": b, "k": rng.binomial(1000, p), "n": 1000}
                for a, p in (("a1", 0.01), ("a2", 0.05)) for b in ("b1", "b2") for _ in range(5)]
        wald = S.logistic(pd.DataFrame(rows), ["A", "B"], "k", "n")
        self.assertTrue(wald.loc["A", "significant"])
        self.assertFalse(wald.loc["B", "significant"])
        flat = pd.DataFrame([{"A": "a1", "B": "b1", "k": 0, "n": 10}, {"A": "a2", "B": "b1", "k": 0, "n": 10}])
        self.assertIsNone(S.logistic(flat, ["A", "B"], "k", "n"))


class TestCompleteDesign(unittest.TestCase):
    def design_b(self):
        """model x mem x alloc as experiment B: dscnn has no value in SRAM (allocation failure)."""
        rows = []
        for model in ("dnn", "cnn", "dscnn"):
            for mem in ("sram", "psram"):
                for alloc in ("static", "dynamic"):
                    for r in range(3):
                        y = np.nan if (model, mem) == ("dscnn", "sram") else 10.0 + r
                        rows.append({"model": model, "mem": mem, "alloc": alloc, "y": y})
        df = pd.DataFrame(rows)
        for f in ("model", "mem", "alloc"):
            df[f] = pd.Categorical(df[f], categories=list(dict.fromkeys(df[f])))
        return df

    def test_drops_the_level_that_empties_cells(self):
        data, factors, dropped = S.complete_design(self.design_b(), ["model", "mem", "alloc"], ["y"])
        self.assertEqual(dropped, [("model", "dscnn")])
        self.assertEqual(factors, ["model", "mem", "alloc"])
        self.assertEqual(len(data), 24)
        self.assertNotIn("dscnn", list(data["model"].cat.categories))
        S.anova(data, factors, "y")  # estimable now

    def test_complete_design_is_untouched(self):
        df = factorial()
        data, factors, dropped = S.complete_design(df, ["A", "B"], ["y"])
        self.assertEqual((len(data), factors, dropped), (len(df), ["A", "B"], []))

    def test_factor_left_with_one_level_is_removed(self):
        df = factorial(reps=2)
        df.loc[df.A == "a1", "y"] = np.nan
        data, factors, dropped = S.complete_design(df, ["A", "B"], ["y"])
        self.assertEqual(dropped, [("A", "a1")])
        self.assertEqual(factors, ["B"])

if __name__ == "__main__":
    unittest.main()
