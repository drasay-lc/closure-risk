"""Score Lewis & Clark and its peers.

1. Current risk: models trained on every institution-year with an observed outcome,
   applied to t = 2025 (latest IPEDS data, fall 2023 / FY2023, is the two-year lag).
2. Risk history: 5-fold cross-fitted predictions grouped by institution, so each
   college's historical score comes from a model that never saw that college.
3. Distress outlook: XGBoost (all controls) for each of the paper's four distress
   indicators, predicting whether the flag is on in t.

Percentiles compare each college with all private nonprofit four-year colleges scored for the same t.

Writes output/peer_scores_current.csv, output/peer_risk_history.csv,
       output/peer_distress_outlook.csv, output/peer_metrics.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from features import DISTRESS, LAST_DATA_YEAR, SCORE_YEAR
from modeling import fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PROC = ROOT / "data" / "processed"
TMAX = {"closed_t": LAST_DATA_YEAR, "closed_3yr": LAST_DATA_YEAR - 2}


def peers() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "config" / "peers.csv")
    return p


def pct_rank(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentile of col among private nonprofit four-year colleges in the same t (100 = riskiest)."""
    ref = df[(df.forprofit == 0) & (df.four_year == 1)]
    out = pd.Series(np.nan, index=df.index)
    for t, g in df.groupby("t"):
        r = ref.loc[ref.t == t, col].dropna().values
        if len(r):
            out.loc[g.index] = g[col].map(lambda v: (r < v).mean() * 100 if pd.notna(v) else np.nan)
    return out


def main() -> None:
    md = pd.read_parquet(PROC / "model_data.parquet").reset_index(drop=True)
    ent = pd.read_parquet(PROC / "entity_year.parquet")
    pr = peers()
    unit2ope = ent.drop_duplicates("UNITID").set_index("UNITID")["opeid6"]
    pr["opeid6"] = pr.unitid.map(unit2ope)
    peer_ope = set(pr.opeid6.dropna())

    # ---------------- 1. current scores from full-sample models
    cur = md[md.t == SCORE_YEAR].copy()
    for y, tmax in TMAX.items():
        train = md[md.t <= tmax]
        for k in ("B", "F"):
            m = fit(k, train, y)
            cur[f"p_{y}_{k}"] = m.predict(cur)
    for c in [c for c in cur.columns if c.startswith("p_")]:
        cur[c.replace("p_", "pct_")] = pct_rank(cur, c)
    keep = ["opeid6", "t"] + [c for c in cur.columns if c.startswith(("p_", "pct_"))]
    sc = pr.merge(cur[keep], on="opeid6", how="left")
    np4 = cur[(cur.forprofit == 0) & (cur.four_year == 1)]
    bench = {c: np4[c].median() for c in keep if c.startswith("p_")}
    sc.to_csv(OUT / "peer_scores_current.csv", index=False)
    allc = cur[(cur.forprofit == 0) & (cur.four_year == 1)][["opeid6", "UNITID", "INSTNM", "STABBR", "p_closed_3yr_F"]]
    allc = allc.sort_values("p_closed_3yr_F", ascending=False)
    allc["rank"] = range(1, len(allc) + 1)
    allc.to_csv(OUT / "np4_scores_current.csv", index=False)
    pd.Series(bench, name="median_private_nonprofit_4yr").to_csv(OUT / "benchmark_medians.csv")
    print(sc[["short", "peer_group", "p_closed_3yr_F", "pct_closed_3yr_F", "p_closed_t_F", "p_closed_3yr_B"]]
          .sort_values("p_closed_3yr_F", ascending=False).to_string())
    print("median NP4:", bench)

    # ---------------- 2. cross-fitted history
    hist = md[["opeid6", "t", "forprofit", "four_year"]].copy()
    gkf = GroupKFold(n_splits=5)
    for y, tmax in TMAX.items():
        hist[f"p_{y}_F"] = np.nan
        for tr_i, te_i in gkf.split(md, groups=md.opeid6):
            tr = md.iloc[tr_i]
            m = fit("F", tr[tr.t <= tmax], y)
            hist.loc[md.index[te_i], f"p_{y}_F"] = m.predict(md.iloc[te_i])
        hist[f"pct_{y}_F"] = pct_rank(hist, f"p_{y}_F")
    h = hist[hist.opeid6.isin(peer_ope)].merge(pr[["opeid6", "key", "short", "peer_group"]], on="opeid6")
    h.to_csv(OUT / "peer_risk_history.csv", index=False)
    # system-wide yearly medians for context
    ref = hist[(hist.forprofit == 0) & (hist.four_year == 1)].groupby("t")[["p_closed_3yr_F", "p_closed_t_F"]]
    ref.median().add_prefix("median_").join(ref.quantile(0.9).add_prefix("p90_")).to_csv(OUT / "np4_risk_by_year.csv")

    # ---------------- 3. distress outlook
    dist = cur[["opeid6"]].copy()
    for d in DISTRESS:
        y = f"y_{d}"
        train = md[(md.t <= LAST_DATA_YEAR) & md[y].notna()]
        m = fit("F", train, y)
        dist[f"p_{d}"] = m.predict(cur)
    dist = pr[["key", "short", "peer_group", "opeid6"]].merge(dist, on="opeid6", how="left")
    dist.to_csv(OUT / "peer_distress_outlook.csv", index=False)

    # ---------------- 4. the underlying metrics for each peer (latest years)
    cols = ["opeid6", "year", "enr_total", "enr_ug", "staff_total", "rev_total", "exp_total", "rev_tuition",
            "endowment", "net_assets_unres", "debt_ppe", "op_margin", "days_cash", "debt_to_assets",
            "debt_to_ebida", "sh_tuition", "sh_scholarships", "enr_chg", "share_ug"] + DISTRESS
    met = ent[ent.opeid6.isin(peer_ope)][cols].merge(pr[["opeid6", "key", "short", "peer_group"]], on="opeid6")
    met.to_csv(OUT / "peer_metrics.csv", index=False)


if __name__ == "__main__":
    main()
