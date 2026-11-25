"""Track record: would the model have flagged real closures before they happened?

Two checks, both on cross-fitted model F scores (5-fold GroupKFold by institution, so
every college is scored by a model that never saw it, as in score.py step 2):

1. Oregon. Every Oregon private nonprofit college that closed or lost its independence
   since 2006 (config/oregon_closures.csv).
2. U.S. peer closures. Private nonprofit four-year colleges that closed (or were absorbed)
   from 2016 to 2027 and that resemble Lewis & Clark's 33-college comparison set:
     * Carnegie basic class among those in the comparison set (R2, Doctoral/Professional,
       Master's large/medium/small, Baccalaureate arts & sciences or diverse fields)
     * academics: an undergraduate program mix that includes the arts & sciences
       (not "professions focus") and a four-year, mostly full-time undergraduate profile
     * a primarily or highly residential campus (Carnegie size & setting)
     * size: in the last year of finance data before closing, total expenses and
       endowment each at least half of the smallest value in the comparison set (FY2023 $)
   Closures come from the IPEDS directory files (2016-2023) plus a sourced list of
   2023-2027 closures that IPEDS does not record yet (config/closure_events.csv).

A score for outcome year t uses data from fall t-2 / FY t-2 and is the chance of closing
in t, t+1 or t+2. A closure is "flagged" when the college sits in the riskiest tenth
(90th percentile or above) of private nonprofit four-year colleges in any of the scores
whose window covers the closure year (t = C-2 to C).

Writes output/backtest_*.csv and pages/track-record/data.js
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from features import LAST_DATA_YEAR, SCORE_YEAR
from modeling import fit
from score import pct_rank

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, OUT, CFG = ROOT / "data/raw", ROOT / "data/processed", ROOT / "output", ROOT / "config"
PAGE = ROOT / "pages" / "track-record"
TMAX = {"closed_t": LAST_DATA_YEAR, "closed_3yr": LAST_DATA_YEAR - 2}
TOP = 90.0  # riskiest tenth

BASIC = {16: "Doctoral: high research", 17: "Doctoral/professional", 18: "Master's: larger programs",
         19: "Master's: medium programs", 20: "Master's: small programs",
         21: "Baccalaureate: arts & sciences", 22: "Baccalaureate: diverse fields"}
PEER_BASIC = set(BASIC)                      # every class in the comparison set
PEER_IPUG = set(range(6, 18))               # arts & sciences present (excludes professions focus)
PEER_UGPRF = set(range(6, 16))              # four-year, medium or high full-time
PEER_SZSET = {7, 8, 10, 11, 13, 14, 16, 17}  # primarily or highly residential


# ------------------------------------------------------------------ scores
def crossfit_history(md: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    cache = PROC / "crossfit_history.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    hist = md[["opeid6", "t", "INSTNM", "STABBR", "forprofit", "four_year"]].copy()
    gkf = GroupKFold(n_splits=5)
    for y, tmax in TMAX.items():
        hist[f"p_{y}"] = np.nan
        for tr_i, te_i in gkf.split(md, groups=md.opeid6):
            tr = md.iloc[tr_i]
            m = fit("F", tr[tr.t <= tmax], y)
            hist.loc[md.index[te_i], f"p_{y}"] = m.predict(md.iloc[te_i])
        hist[f"pct_{y}"] = pct_rank(hist, f"p_{y}")
    ref = hist[(hist.forprofit == 0) & (hist.four_year == 1)]
    hist["rank_3yr"] = ref.groupby("t").p_closed_3yr.rank(ascending=False, method="min")
    hist["n_np4"] = hist.t.map(ref.groupby("t").size())
    hist.to_parquet(cache)
    return hist


# ------------------------------------------------------------------ carnegie
def carnegie_latest() -> pd.DataFrame:
    """Latest Carnegie fields per UNITID from HD 2016-2023 (2015/2018/2021 editions share codes)."""
    rows = []
    for y in range(2016, LAST_DATA_YEAR + 1):
        hd = pd.read_csv(RAW / str(y) / f"hd{y}.csv", encoding="latin1", low_memory=False)
        hd.columns = [c.upper() for c in hd.columns]
        for ed in ("C21", "C18", "C15"):
            need = [f"{ed}{v}" for v in ("BASIC", "IPUG", "UGPRF", "SZSET")]
            if all(c in hd.columns for c in need):
                d = hd[["UNITID"] + need].copy()
                d.columns = ["UNITID", "basic", "ipug", "ugprf", "szset"]
                d["carnegie_edition"] = "20" + ed[1:]
                d["hd_year"] = y
                rows.append(d)
                break
    c = pd.concat(rows).sort_values(["UNITID", "hd_year"]).groupby("UNITID").tail(1)
    return c.drop(columns="hd_year")


# ------------------------------------------------------------------ summaries
def summarize(h: pd.DataFrame, close_year: float) -> dict:
    """Scores around a closure in year C (t = C-2..C cover it)."""
    C = int(close_year)
    g = h.set_index("t").sort_index()
    g = g[g.rank_3yr.notna()]  # years scored among private nonprofit four-year colleges
    win = g.loc[(g.index >= C - 2) & (g.index <= min(C, SCORE_YEAR))]
    five = g.loc[(g.index >= C - 6) & (g.index <= min(C, SCORE_YEAR))]
    out = {"close_year": C}
    if win.empty:
        return out | {"flagged": "no score"}
    best = win.pct_closed_3yr.idxmax()
    out.update(
        peak_pct=round(win.pct_closed_3yr.max(), 1), peak_t=int(best), peak_rank=int(win.rank_3yr[best]),
        peak_n=int(win.n_np4[best]), peak_risk_pct=round(100 * win.p_closed_3yr[best], 1),
        flagged="yes" if win.pct_closed_3yr.max() >= TOP else "no",
        pct_two_out=round(g.pct_closed_3yr.get(C - 2, np.nan), 1),
    )
    hit = five[five.pct_closed_3yr >= TOP]
    out["years_flagged_6"] = int(len(hit))
    if len(hit) and out["flagged"] == "yes":
        # earliest flag in the six score years up to the closure
        first = int(hit.index.min())
        out.update(first_flag_t=first, first_flag_data_year=first - 2, lead_years=C - first)
    return out


def last_metrics(ent: pd.DataFrame, opeid: str, close_year: int) -> dict:
    """Size = median of the last five fiscal years of finance data before closing (FY2023 $),
    ignoring zero endowments (a college merging away can report 0 in its final year).
    Enrollment, margin and tuition share are from the last year."""
    e = ent[(ent.opeid6 == opeid) & (ent.year < close_year) & ent.exp_total_r.notna()].sort_values("year")
    if e.empty:
        return {}
    r, l5 = e.iloc[-1], e.tail(5)
    endow = l5.endowment_r[l5.endowment_r > 0]
    return {"metrics_fy": int(r.year), "size_fys": f"{int(l5.year.min())}-{int(l5.year.max())}",
            "enrollment": r.enr_total, "expenses_m": round(l5.exp_total_r.median() / 1e6, 1),
            "endowment_m": round(endow.median() / 1e6, 1) if len(endow) else 0.0,
            "margin_pct": round(100 * r.op_margin, 1) if pd.notna(r.op_margin) else np.nan,
            "tuition_share_pct": round(100 * r.sh_tuition, 1) if pd.notna(r.sh_tuition) else np.nan,
            "UNITID": int(r.UNITID)}


def main() -> None:
    md = pd.read_parquet(PROC / "model_data.parquet").reset_index(drop=True)
    ent = pd.read_parquet(PROC / "entity_year.parquet")
    hist = crossfit_history(md)
    keep_h = ["opeid6", "t", "p_closed_3yr", "pct_closed_3yr", "rank_3yr", "n_np4"]
    unit2ope = ent.drop_duplicates("UNITID").set_index("UNITID")["opeid6"]
    name = ent.sort_values("year").groupby("opeid6").INSTNM.last()
    state = ent.sort_values("year").groupby("opeid6").STABBR.last()
    car = carnegie_latest()

    # ---------------- comparison set (L&C + 33 peers) for the size thresholds
    pr = pd.read_csv(CFG / "peers.csv")
    pr["opeid6"] = pr.unitid.map(unit2ope)
    cs = ent[ent.opeid6.isin(pr.opeid6) & (ent.year == LAST_DATA_YEAR)]
    exp_min, end_min = cs.exp_total_r.min() / 1e6, cs.endowment_r.min() / 1e6
    exp_floor, end_floor = exp_min / 2, end_min / 2
    lc_ope = pr.loc[pr.key == "lewis-clark", "opeid6"].iloc[0]

    # ---------------- 1. Oregon
    orc = pd.read_csv(CFG / "oregon_closures.csv", dtype={"opeid6": str})
    o_rows, o_hist = [], []
    for _, r in orc.iterrows():
        row = r.to_dict()
        if isinstance(r.opeid6, str) and r.opeid6 in set(hist.opeid6):
            h = hist[hist.opeid6 == r.opeid6]
            row |= summarize(h, r.close_year) | last_metrics(ent, r.opeid6, int(r.close_year))
            o_hist.append(h[keep_h].assign(institution=r.institution))
        o_rows.append(row)
    oregon = pd.DataFrame(o_rows)
    oregon.to_csv(OUT / "backtest_oregon.csv", index=False)
    # Oregon colleges still open, for the "flagged but open" list
    o_open = hist[(hist.STABBR == "OR") & (hist.forprofit == 0) & (hist.four_year == 1)
                  & ~hist.opeid6.isin(orc.opeid6.dropna())
                  & ~hist.opeid6.isin(ent[ent.close_year.notna()].opeid6)]
    o_open = o_open[o_open.t >= 2016]
    o_open_flags = (o_open[o_open.pct_closed_3yr >= TOP].groupby("opeid6").t.agg(list)
                    .rename("years_in_riskiest_tenth").reset_index())
    o_open_flags["institution"] = o_open_flags.opeid6.map(name)
    o_open_flags["pct_2025"] = o_open_flags.opeid6.map(
        o_open[o_open.t == SCORE_YEAR].set_index("opeid6").pct_closed_3yr.round(1))
    o_open_flags.to_csv(OUT / "backtest_oregon_open_flagged.csv", index=False)

    # ---------------- 2. U.S. closures 2016-2027 and the peer screen
    ev = pd.read_csv(CFG / "closure_events.csv")
    ev["opeid6"] = ev.unitid.map(unit2ope)
    np4_ent = ent[(ent.forprofit == 0) & (ent.four_year == 1)]
    cy = np4_ent.groupby("opeid6").close_year.max()
    ipeds = cy[(cy >= 2016) & (cy <= LAST_DATA_YEAR)].rename("close_year").reset_index()
    ipeds = ipeds[~ipeds.opeid6.isin(ev.opeid6)]
    ipeds = ipeds.assign(institution=ipeds.opeid6.map(name), state=ipeds.opeid6.map(state),
                         event="Closure recorded in the IPEDS directory", event_type="closed",
                         source="IPEDS directory files (HD), via closures.csv")
    pool = pd.concat([ipeds, ev[ev.opeid6.notna()].drop(columns="unitid")], ignore_index=True)
    pool = pool[pool.opeid6.isin(np4_ent.opeid6)]

    rows = []
    for _, r in pool.iterrows():
        row = r.to_dict() | last_metrics(ent, r.opeid6, int(r.close_year))
        if "UNITID" not in row:
            continue
        c = car[car.UNITID == row["UNITID"]]
        c = c.iloc[0] if len(c) else pd.Series(dtype=float)
        row |= {k: c.get(k, np.nan) for k in ("basic", "ipug", "ugprf", "szset", "carnegie_edition")}
        row["carnegie_class"] = BASIC.get(row["basic"], "other")
        row["c_carnegie"] = row["basic"] in PEER_BASIC
        row["c_academics"] = (row["ipug"] in PEER_IPUG) and (row["ugprf"] in PEER_UGPRF)
        row["c_residential"] = row["szset"] in PEER_SZSET
        row["c_size"] = (row["expenses_m"] >= exp_floor) and (row.get("endowment_m", 0) or 0) >= end_floor
        profile = all(row[k] for k in ("c_carnegie", "c_academics", "c_residential"))
        row["tier"] = ("peer" if profile and row["c_size"]
                       else "near peer" if profile and row["expenses_m"] >= exp_floor / 2
                       else "not a peer")
        row |= summarize(hist[hist.opeid6 == r.opeid6], r.close_year)
        rows.append(row)
    us = pd.DataFrame(rows).sort_values(["tier", "close_year", "institution"])
    us.to_csv(OUT / "backtest_us_candidates.csv", index=False)
    peers_us = us[us.tier != "not a peer"]
    peers_us.to_csv(OUT / "backtest_us_peers.csv", index=False)

    # ---------------- histories for the page
    ush = hist[hist.opeid6.isin(peers_us.opeid6)][keep_h].merge(
        peers_us[["opeid6", "institution"]], on="opeid6")
    lch = hist[hist.opeid6 == lc_ope][keep_h].assign(institution="Lewis & Clark College")
    hh = pd.concat(o_hist + [ush, lch])
    hh.to_csv(OUT / "backtest_histories.csv", index=False)

    # context: how often the riskiest tenth closes, private nonprofit 4-yr, t <= 2021
    lab = md[(md.forprofit == 0) & (md.four_year == 1) & (md.t <= TMAX["closed_3yr"])][["opeid6", "t", "closed_3yr"]]
    lab = lab.merge(hist[["opeid6", "t", "pct_closed_3yr"]], on=["opeid6", "t"])
    top = lab[lab.pct_closed_3yr >= TOP]
    context = {
        "top_tenth_close_rate_pct": round(100 * top.closed_3yr.mean(), 1),
        "rest_close_rate_pct": round(100 * lab[lab.pct_closed_3yr < TOP].closed_3yr.mean(), 2),
        "exp_min_m": round(exp_min, 1), "end_min_m": round(end_min, 1),
        "exp_floor_m": round(exp_floor, 1), "end_floor_m": round(end_floor, 1),
        "lc_pct_2025": round(float(hist[(hist.opeid6 == lc_ope) & (hist.t == SCORE_YEAR)].pct_closed_3yr.iloc[0]), 1),
        "n_us_closures": int(len(us)), "score_year": SCORE_YEAR, "last_data_year": LAST_DATA_YEAR,
        "oregon_tested": int(oregon.flagged.isin(["yes", "no"]).sum()),
        "oregon_flagged": int((oregon.flagged == "yes").sum()),
    }
    for tier, g in us.groupby("tier"):
        k = tier.replace(" ", "_")
        context[f"{k}_n"] = int(len(g))
        context[f"{k}_scored"] = int(g.flagged.isin(["yes", "no"]).sum())
        context[f"{k}_flagged"] = int((g.flagged == "yes").sum())
    cmp_set = cs[["opeid6", "INSTNM", "exp_total_r", "endowment_r"]].assign(
        expenses_m=lambda d: (d.exp_total_r / 1e6).round(1), endowment_m=lambda d: (d.endowment_r / 1e6).round(1))
    pd.Series(context).to_csv(OUT / "backtest_context.csv", header=["value"])

    # ---------------- page data
    PAGE.mkdir(parents=True, exist_ok=True)
    def recs(df):
        return json.loads(df.to_json(orient="records"))
    hcols = ["institution", "t", "pct_closed_3yr", "rank_3yr", "n_np4", "p_closed_3yr"]
    payload = {
        "context": context,
        "oregon": recs(oregon), "oregon_open": recs(o_open_flags),
        "us": recs(us), "compare": recs(cmp_set[["INSTNM", "expenses_m", "endowment_m"]]),
        "hist": recs(hh[hcols].assign(pct_closed_3yr=lambda d: d.pct_closed_3yr.round(1),
                                      p_closed_3yr=lambda d: (100 * d.p_closed_3yr).round(2))),
    }
    (PAGE / "data.js").write_text("window.TRACK = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    for f in ("backtest_oregon.csv", "backtest_us_candidates.csv", "backtest_histories.csv"):
        (PAGE / f).write_bytes((OUT / f).read_bytes())
    print(oregon[["institution", "close_year", "peak_pct", "peak_t", "flagged", "lead_years"]].to_string())
    print(peers_us[["institution", "state", "close_year", "tier", "expenses_m", "endowment_m", "carnegie_class",
                    "peak_pct", "peak_t", "flagged", "lead_years"]].to_string())
    print(context)


if __name__ == "__main__":
    main()
