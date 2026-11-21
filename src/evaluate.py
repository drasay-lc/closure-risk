"""Estimate models A-F on a 75% training split and evaluate on the held-out 25%.

The split is by institution (OPEID), so no college appears in both sets.
Outcomes: closed_t (closure in year t, t <= 2023) and closed_3yr (closure in t..t+2, t <= 2021).

Writes to output/:
  model_auc.csv            AUC by model, outcome, and for private nonprofit four-year colleges only
  deciles_<outcome>.csv    actual closure share by predicted-risk decile (model F and C)
  topn_<outcome>.csv       closures among the N highest-risk institution-years
  tpr_fpr_<outcome>.csv    false positive rates at target true positive rates
  importance.csv           XGBoost gain by variable family; LPM standardized coefficients
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from features import LAST_DATA_YEAR
from modeling import MODELS, SEED, base_name, fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUTCOMES = {"closed_t": LAST_DATA_YEAR, "closed_3yr": LAST_DATA_YEAR - 2}


def auc(y, p):
    ok = ~np.isnan(p)
    if y[ok].nunique() < 2:
        return np.nan
    return roc_auc_score(y[ok], p[ok])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    md = pd.read_parquet(ROOT / "data" / "processed" / "model_data.parquet")
    rows, imp_rows = [], []
    for y, tmax in OUTCOMES.items():
        d = md[(md.t <= tmax)].reset_index(drop=True)
        tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(d, groups=d.opeid6))
        tr, te = d.iloc[tr_idx], d.iloc[te_idx]
        np4 = (te.forprofit == 0) & (te.four_year == 1)
        preds = {}
        for k in MODELS:
            m = fit(k, tr, y)
            p = m.predict(te)
            preds[k] = p
            ok = ~np.isnan(p)
            rows.append(dict(outcome=y, model=k, description=MODELS[k],
                             auc=auc(te[y], p), auc_np4=auc(te[y][np4], p[np4.values]),
                             eval_obs=int(ok.sum()), eval_closures=int(te[y][ok].sum()),
                             train_obs=len(tr), train_closures=int(tr[y].sum())))
            print(rows[-1])
            if k in ("D", "E", "F"):
                g = pd.Series(m.model.get_booster().get_score(importance_type="gain"))
                g = g.groupby(g.index.map(base_name)).sum()
                g = g / g.sum()
                imp_rows += [dict(outcome=y, model=k, variable=v, importance=s) for v, s in g.items()]
            if k in ("A", "B"):
                res = m.model
                X = tr[m.selected]
                std = X.std()
                for v in m.selected:
                    imp_rows.append(dict(outcome=y, model=k, variable=v, importance=res.params[v] * std[v],
                                         pvalue=res.pvalues[v]))

        # decile table (paper Table 5 style)
        dec = []
        for k in ("C", "F"):
            p = pd.Series(preds[k], index=te.index)
            q = pd.qcut(p.rank(method="first"), 10, labels=range(1, 11))
            t = te.assign(decile=q).groupby("decile", observed=True)[y].agg(["size", "sum", "mean"])
            t["mean_pred"] = p.groupby(q, observed=True).mean()
            t["model"] = k
            dec.append(t.reset_index())
        pd.concat(dec).to_csv(OUT / f"deciles_{y}.csv", index=False)

        # top-N
        top = []
        for k, p in preds.items():
            s = pd.Series(p, index=te.index).dropna().sort_values(ascending=False)
            for n in (100, 500):
                top.append(dict(model=k, top_n=n, closures=int(te.loc[s.index[:n], y].sum()),
                                share=te.loc[s.index[:n], y].mean()))
        pd.DataFrame(top).to_csv(OUT / f"topn_{y}.csv", index=False)

        # TPR / FPR at target true-positive rates (paper Table 7)
        tf = []
        for k, p in preds.items():
            ok = ~np.isnan(p)
            fpr, tpr, thr = roc_curve(te[y][ok], p[ok])
            for target in (0.5, 0.6, 0.7, 0.8, 0.9):
                i = int(np.argmax(tpr >= target))
                tf.append(dict(model=k, target_tpr=target, tpr=tpr[i], fpr=fpr[i], threshold=thr[i]))
        pd.DataFrame(tf).to_csv(OUT / f"tpr_fpr_{y}.csv", index=False)

    pd.DataFrame(rows).to_csv(OUT / "model_auc.csv", index=False)
    pd.DataFrame(imp_rows).to_csv(OUT / "importance.csv", index=False)
    print(pd.DataFrame(rows)[["outcome", "model", "auc", "auc_np4", "eval_obs", "eval_closures"]].to_string())


if __name__ == "__main__":
    main()
