"""Shape the model outputs into the small CSVs the results dashboard reads (dashboard/data/)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
DD = ROOT / "dashboard" / "data"
PAPER_AUC = {  # Kelchen, Ritter & Webber (2024), evaluation-sample AUCs (percent)
    ("closed_t", "A"): 78.7, ("closed_t", "B"): 83.4, ("closed_t", "C"): 75.6,
    ("closed_t", "D"): 76.9, ("closed_t", "E"): 80.6, ("closed_t", "F"): 81.8,
    ("closed_3yr", "F"): 86.8,
}
SCEN = {"baseline": "Flat enrollment", "persist": "2019-23 decline persists",
        "cliff_onetime": "15% drop in 2025", "cliff_gradual": "15% gradual decline by 2029",
        "cliff_gradual_fixed50": "15% gradual decline, half of costs fixed"}


def main() -> None:
    DD.mkdir(parents=True, exist_ok=True)
    pr = pd.read_csv(ROOT / "config" / "peers.csv")
    lc = "lewis-clark"

    # current scores
    sc = pd.read_csv(OUT / "peer_scores_current.csv")
    cur = pd.DataFrame({
        "college": sc.short, "group": sc.peer_group,
        "risk_3yr_pct": (sc.p_closed_3yr_F * 100).round(2),
        "percentile": sc.pct_closed_3yr_F.round(0).astype(int),
        "risk_this_year_pct": (sc.p_closed_t_F * 100).round(3),
    }).sort_values("risk_3yr_pct", ascending=False)
    cur.to_csv(DD / "current_risk.csv", index=False)

    allc = pd.read_csv(OUT / "np4_scores_current.csv")
    bench = pd.DataFrame([
        {"key": "np4_median_3yr", "measure": "Median private nonprofit four-year college, 3-year closure risk (%)",
         "value": round(allc.p_closed_3yr_F.median() * 100, 2)},
        {"key": "np4_count", "measure": "Private nonprofit four-year colleges scored", "value": len(allc)},
    ])
    bench.to_csv(DD / "benchmarks.csv", index=False)

    # history: L&C and group medians of the percentile
    h = pd.read_csv(OUT / "peer_risk_history.csv")
    h = h[h.t <= 2025]
    lcs = h[h.key == lc][["t", "pct_closed_3yr_F"]].assign(series="Lewis & Clark")
    grp = (h[h.key != lc].groupby(["t", "peer_group"]).pct_closed_3yr_F.median().reset_index()
             .rename(columns={"peer_group": "series"}))
    grp["series"] = grp.series + " median"
    hist = pd.concat([lcs, grp])[["t", "series", "pct_closed_3yr_F"]]
    hist.columns = ["year", "series", "percentile"]
    hist["percentile"] = hist.percentile.round(1)
    hist.sort_values(["series", "year"]).to_csv(DD / "risk_history.csv", index=False)

    # distress outlook
    d = pd.read_csv(OUT / "peer_distress_outlook.csv")
    dist = pd.DataFrame({
        "college": d.short, "group": d.peer_group,
        "revenue_down_10_pct": (d.p_d_rev_down10 * 100).round(0).astype(int),
        "neg_margin_3_of_5_pct": (d.p_d_neg_margin_3of5 * 100).round(0).astype(int),
        "enrollment_down_10_pct": (d.p_d_enr_down10 * 100).round(0).astype(int),
        "enrollment_slide_3yr_pct": (d.p_d_enr_down5_3yrs * 100).round(0).astype(int),
    })
    dist["flags_over_half"] = (dist.iloc[:, 2:] >= 50).sum(axis=1)
    dist.sort_values(["flags_over_half", "enrollment_down_10_pct"], ascending=False).to_csv(
        DD / "distress_outlook.csv", index=False)

    # 2023 metrics behind the scores
    m = pd.read_csv(OUT / "peer_metrics.csv")
    m = m[m.year == 2023]
    met = pd.DataFrame({
        "college": m.short, "group": m.peer_group,
        "fall_2023_enrollment": m.enr_total.astype(int),
        "fy2023_margin_pct": (m.op_margin * 100).round(1),
        "tuition_share_pct": (m.sh_tuition * 100).round(0),
        "days_cash_proxy": m.days_cash.round(0),
        "endowment_m": (m.endowment / 1e6).round(0),
        "plant_debt_m": (m.debt_ppe / 1e6).round(0),
    }).sort_values("fy2023_margin_pct")
    met.to_csv(DD / "metrics_2023.csv", index=False)

    # scenarios, 2031
    s = pd.read_csv(OUT / "scenario_peers.csv")
    s = s[s.t == 2031]
    sp = s.pivot_table(index=["short", "peer_group"], columns="scenario", values="p_closed_3yr_F") * 100
    sp = sp.round(2).reset_index().rename(columns={"short": "college", "peer_group": "group"})
    sp["fixed_cost_change"] = (sp.cliff_gradual_fixed50 - sp.baseline).round(2)
    sp.sort_values("cliff_gradual_fixed50", ascending=False).to_csv(DD / "scenarios_2031.csv", index=False)
    med = s.groupby("scenario").np4_median_3yr_F.first().mul(100).round(2)
    pd.DataFrame({"scenario": med.index, "label": med.index.map(SCEN), "np4_median_pct": med.values}).to_csv(
        DD / "scenario_medians_2031.csv", index=False)
    e = pd.read_csv(OUT / "scenario_expected_closures.csv")
    e = e[e.t.between(2027, 2031)].groupby("scenario")[["p_closed_t_B", "p_closed_t_F"]].mean().round(0)
    e = e.reset_index()
    e["label"] = e.scenario.map(SCEN)
    e.columns = ["scenario", "closures_per_year_lasso", "closures_per_year_xgboost", "label"]
    e.to_csv(DD / "system_closures_2027_2031.csv", index=False)

    # model check
    a = pd.read_csv(OUT / "model_auc.csv")
    rows = []
    for k in "ABCDEF":
        r = {"model": k, "description": a[(a.outcome == "closed_t") & (a.model == k)].description.iloc[0]}
        for y, lab in (("closed_t", "this_year"), ("closed_3yr", "three_year")):
            x = a[(a.outcome == y) & (a.model == k)].iloc[0]
            r[f"auc_{lab}"] = round(x.auc * 100, 1)
            r[f"auc_{lab}_np4"] = round(x.auc_np4 * 100, 1)
            r[f"paper_auc_{lab}"] = PAPER_AUC.get((y, k))
        rows.append(r)
    pd.DataFrame(rows).to_csv(DD / "model_check.csv", index=False)

    # out-of-sample: later closures
    lat = pd.read_csv(ROOT / "config" / "later_closures.csv")
    n = len(allc)
    oos = []
    for _, r in lat.iterrows():
        hit = allc[allc.INSTNM.str.match(r.name_pattern)]
        if len(hit) == 1:
            k = int(hit["rank"].iloc[0])
            oos.append({"institution": r.institution, "event": r.event, "rank": k, "of": n,
                        "top_decile": "Yes" if k <= n / 10 else "No",
                        "risk_3yr_pct": round(float(hit.p_closed_3yr_F.iloc[0]) * 100, 1)})
    pd.DataFrame(oos).sort_values("rank").to_csv(DD / "later_closures_check.csv", index=False)
    print(sorted(p.name for p in DD.iterdir()))


if __name__ == "__main__":
    main()
