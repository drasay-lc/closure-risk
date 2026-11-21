"""Build the modelling dataset following Kelchen, Ritter & Webber (2024), Philadelphia Fed WP 24-20.

Steps
 1. Keep private nonprofit and for-profit Title IV institutions (valid OPEID).
 2. Aggregate UNITIDs to the 6-digit OPEID (the paper's unit), coding an OPEID as
    four-year if any member is four-year.
 3. Date closures at the OPEID level (main campus, OPEID ending "00").
 4. Construct the paper's enrollment, staff, revenue, expense, asset/debt and
    derived financial metrics, plus its four distress indicators.
 5. Lag features by 2-5 years and attach outcomes: closure in t, closure in t..t+2,
    and each distress indicator in t.

Outputs
  data/processed/entity_year.parquet   one row per OPEID6 x year with contemporaneous features
  data/processed/model_data.parquet    one row per OPEID6 x outcome year t with lagged features
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
LAST_DATA_YEAR = 2023          # last IPEDS collection year available
SCORE_YEAR = LAST_DATA_YEAR + 2  # t for which the latest data are the 2-year lag

LEVEL_COLS = ["enr_total", "enr_ug", "enr_ft", "staff_total", "staff_ft", "staff_instr",
              "rev_total", "rev_tuition", "rev_aux", "rev_invest", "rev_gifts_grants",
              "exp_total", "exp_instruction", "exp_scholarships", "exp_salaries",
              "exp_depreciation", "exp_interest", "assets_total", "liab_total", "debt_ppe",
              "net_assets_unres", "net_assets_temp", "ppe_net", "endowment"]
DOLLAR_COLS = [c for c in LEVEL_COLS if not c.startswith(("enr_", "staff_"))]

# features entering the models (computed for each entity-year, then lagged)
FEATURES = [
    # enrollment
    "ln_enr", "enr_chg", "share_ft", "share_ug",
    # staff
    "ln_staff", "staff_chg", "share_ft_staff", "share_instr_staff",
    # revenue
    "ln_rev", "rev_chg", "sh_tuition", "sh_aux", "sh_invest", "sh_gifts_grants",
    # expenses
    "ln_exp", "exp_chg", "sh_instruction", "sh_scholarships", "sh_interest", "sh_depreciation", "sh_salaries",
    # assets and debt
    "slog_unres_na", "ln_debt", "ln_endow",
    # derived financial metrics
    "op_margin", "op_margin_chg", "days_cash", "days_cash_chg", "slog_ebida", "debt_to_ebida",
    "debt_to_assets", "debt_to_assets_chg",
    # derived distress indicators
    "d_rev_down10", "d_neg_margin_3of5", "d_enr_down10", "d_enr_down5_3yrs",
]
DISTRESS = ["d_rev_down10", "d_neg_margin_3of5", "d_enr_down10", "d_enr_down5_3yrs"]
STATIC = ["forprofit", "four_year", "two_year"]


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load_cpi() -> pd.Series:
    cpi = pd.read_csv(ROOT / "config" / "cpi.csv").set_index("fiscal_year")["fiscal_year_cpi"]
    return cpi.loc[LAST_DATA_YEAR] / cpi  # factor to FY2023 dollars


def entity_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_parquet(PROC / "panel.parquet")
    p = p[p.CONTROL.isin([2, 3])].copy()
    op = pd.to_numeric(p["OPEID"], errors="coerce")
    p = p[op > 0].copy()
    p["opeid6"] = p["OPEID"].str[:6]
    p["is_main"] = p["OPEID"].str[-2:] == "00"
    p["four"] = (p["ICLEVEL"] == 1).astype(int)

    # closure dating at the OPEID level -------------------------------------
    hd_years = p[p.year >= 2002]
    last_seen = hd_years.groupby("opeid6")["year"].max()
    first_seen = hd_years.groupby("opeid6")["year"].min()
    mains = hd_years[hd_years.is_main].sort_values("year").groupby("opeid6").tail(1)
    main_close = mains.set_index("opeid6")["close_year"]
    main_merged = mains.set_index("opeid6")["merged"].astype(bool)
    ent = pd.DataFrame({"first_year": first_seen, "last_seen": last_seen})
    ent["main_close"] = main_close
    ent["merged"] = main_merged.reindex(ent.index).fillna(False).astype(bool)
    vanished = ent.last_seen < LAST_DATA_YEAR
    # ignore a main-campus closure date if the OPEID kept reporting afterwards (re-IDs, branch swaps)
    ent.loc[ent.last_seen > ent.main_close, "main_close"] = np.nan
    ent["close_year"] = np.where(ent.main_close.between(2002, LAST_DATA_YEAR + 1), ent.main_close,
                                 np.where(vanished, ent.last_seen + 1, np.nan))
    ent.loc[ent.merged, "close_year"] = np.nan
    ent.loc[ent.close_year > LAST_DATA_YEAR + 1, "close_year"] = np.nan

    # aggregate to OPEID6 x year --------------------------------------------
    agg = {c: (lambda s: s.sum(min_count=1)) for c in LEVEL_COLS}
    num = p.groupby(["opeid6", "year"])[LEVEL_COLS].sum(min_count=1)
    attrs = (p.sort_values(["is_main"], ascending=False)
               .groupby(["opeid6", "year"])
               .agg(UNITID=("UNITID", "first"), INSTNM=("INSTNM", "first"), STABBR=("STABBR", "first"),
                    CONTROL=("CONTROL", "first"), four_year=("four", "max"),
                    two_year=("ICLEVEL", lambda s: int((s == 2).any())), n_units=("UNITID", "nunique")))
    e = attrs.join(num).reset_index()
    e["two_year"] = np.where(e.four_year == 1, 0, e.two_year)
    e["forprofit"] = (e.CONTROL == 3).astype(int)
    e = e.merge(ent[["close_year", "first_year"]], left_on="opeid6", right_index=True, how="left")
    return e, ent


def add_features(e: pd.DataFrame, cpi: pd.Series | None = None) -> pd.DataFrame:
    e = e.sort_values(["opeid6", "year"]).copy()
    f = load_cpi() if cpi is None else cpi
    real = e["year"].map(f)
    for c in DOLLAR_COLS:
        e[c + "_r"] = e[c] * real
    g = e.groupby("opeid6", group_keys=False)

    def pct(c):  # nominal year-over-year % change, only across consecutive years
        prev = g[c].shift(1)
        gap = e["year"] - g["year"].shift(1)
        out = (e[c] - prev) / prev.abs()
        return out.where((gap == 1) & (prev.abs() > 0))

    pos = lambda s: s.where(s > 0)
    e["ln_enr"] = np.log(pos(e.enr_total))
    e["enr_chg"] = pct("enr_total")
    e["share_ft"] = e.enr_ft / pos(e.enr_total)
    e["share_ug"] = e.enr_ug / pos(e.enr_total)
    e["ln_staff"] = np.log(pos(e.staff_total))
    e["staff_chg"] = pct("staff_total")
    e["share_ft_staff"] = e.staff_ft / pos(e.staff_total)
    e["share_instr_staff"] = e.staff_instr / pos(e.staff_total)

    rev = pos(e.rev_total)
    e["ln_rev"] = np.log(pos(e.rev_total_r))
    e["rev_chg"] = pct("rev_total")
    e["sh_tuition"] = e.rev_tuition / rev
    e["sh_aux"] = e.rev_aux / rev
    e["sh_invest"] = e.rev_invest / rev
    e["sh_gifts_grants"] = e.rev_gifts_grants / rev
    exp = pos(e.exp_total)
    e["ln_exp"] = np.log(pos(e.exp_total_r))
    e["exp_chg"] = pct("exp_total")
    e["sh_instruction"] = e.exp_instruction / exp
    e["sh_scholarships"] = e.exp_scholarships / exp
    e["sh_interest"] = e.exp_interest / exp
    e["sh_depreciation"] = e.exp_depreciation / exp
    e["sh_salaries"] = e.exp_salaries / exp

    e["slog_unres_na"] = slog(e.net_assets_unres_r)
    e["ln_debt"] = np.log1p(e.debt_ppe_r.clip(lower=0))
    e["ln_endow"] = np.log1p(e.endowment_r.clip(lower=0))

    # derived metrics (IPEDS has no cash line; see README for the proxies used)
    e["op_margin"] = (e.rev_total - e.exp_total) / rev
    e["op_margin_chg"] = e.op_margin - g["op_margin"].shift(1)
    cash_exp_day = (e.exp_total - e.exp_depreciation.fillna(0)) / 365
    e["days_cash"] = e.net_assets_unres / cash_exp_day.where(cash_exp_day > 0)
    e["days_cash_chg"] = e.days_cash - g["days_cash"].shift(1)
    ebida = e.rev_total - e.exp_total + e.exp_interest + e.exp_depreciation
    e["slog_ebida"] = slog(ebida * real)
    d2e = e.debt_ppe / ebida.where(ebida > 0)
    d2e = d2e.where(~((ebida <= 0) & (e.debt_ppe > 0)), 50.0)   # debt with no EBIDA -> very high
    d2e = d2e.where(~(e.debt_ppe == 0), 0.0)
    e["debt_to_ebida"] = d2e
    e["debt_to_assets"] = e.liab_total / pos(e.assets_total)
    e["debt_to_assets_chg"] = e.debt_to_assets - g["debt_to_assets"].shift(1)

    # distress indicators (paper Section III) -------------------------------
    def roll_max(c, w=5):
        return g[c].transform(lambda s: s.rolling(w, min_periods=1).max())

    def flag(cond, *inputs):
        out = cond.astype(float)
        for i in inputs:
            out = out.where(i.notna())
        return out

    e["d_rev_down10"] = flag(e.rev_total_r < 0.9 * roll_max("rev_total_r"), e.rev_total_r)
    neg = (e.op_margin < 0).astype(float).where(e.op_margin.notna())
    nneg = neg.groupby(e.opeid6).transform(lambda s: s.rolling(5, min_periods=3).sum())
    e["d_neg_margin_3of5"] = (nneg >= 3).astype(float).where(nneg.notna())
    e["d_enr_down10"] = flag(e.enr_total < 0.9 * roll_max("enr_total"), e.enr_total)
    dn = (e.enr_chg <= -0.05).astype(float).where(e.enr_chg.notna())
    n3 = dn.groupby(e.opeid6).transform(lambda s: s.rolling(3, min_periods=3).sum())
    e["d_enr_down5_3yrs"] = (n3 == 3).astype(float).where(n3.notna())
    return e


def winsorize(df: pd.DataFrame, cols, q=0.025, ref=None) -> pd.DataFrame:
    ref = df if ref is None else ref
    for c in cols:
        if c.startswith("d_"):
            continue
        lo, hi = ref[c].quantile([q, 1 - q])
        df[c] = df[c].clip(lo, hi)
    return df


def model_data(e: pd.DataFrame, max_lag: int = 5, t_last: int = SCORE_YEAR) -> pd.DataFrame:
    """One row per entity and outcome year t, with features from t-2 ... t-max_lag."""
    e = e.set_index(["opeid6", "year"])
    base = e.reset_index()[["opeid6", "year"]]
    rows = []
    for t in range(2004, t_last + 1):
        src = base[base.year == t - 2].copy()      # must be observed at t-2
        src["t"] = t
        rows.append(src[["opeid6", "t"]])
    md = pd.concat(rows, ignore_index=True)
    static = e.reset_index().sort_values("year").groupby("opeid6").tail(1).set_index("opeid6")
    for lag in range(2, max_lag + 1):
        key = pd.MultiIndex.from_arrays([md.opeid6, md.t - lag])
        vals = e.reindex(key)[FEATURES].reset_index(drop=True)
        vals.columns = [f"{c}_L{lag}" for c in FEATURES]
        md = pd.concat([md.reset_index(drop=True), vals], axis=1)
    md = md.copy()
    # time-invariant characteristics, taken from the t-2 record
    key2 = pd.MultiIndex.from_arrays([md.opeid6, md.t - 2])
    s2 = e.reindex(key2)[["UNITID", "INSTNM", "STABBR", "forprofit", "four_year", "two_year"]].reset_index(drop=True)
    md = pd.concat([md, s2], axis=1)
    md["close_year"] = md.opeid6.map(static["close_year"])
    # drop entity-years after closure
    md = md[~(md.close_year < md.t)].copy()
    md["closed_t"] = (md.close_year == md.t).astype(int)
    md["closed_3yr"] = md.close_year.between(md.t, md.t + 2).astype(int)
    # distress outcomes measured in t (contemporaneous flags)
    keyt = pd.MultiIndex.from_arrays([md.opeid6, md.t])
    yt = e.reindex(keyt)[DISTRESS].reset_index(drop=True)
    yt.columns = [f"y_{c}" for c in DISTRESS]
    md = pd.concat([md.reset_index(drop=True), yt], axis=1)
    return md


def main() -> None:
    e, ent = entity_panel()
    e = add_features(e)
    e.to_parquet(PROC / "entity_year.parquet", index=False)
    feat_cols = [c for c in e.columns if c in FEATURES]
    md = model_data(e)
    lagged = [c for c in md.columns if c.rsplit("_L", 1)[0] in FEATURES and "_L" in c]
    md = winsorize(md, lagged)
    md.to_parquet(PROC / "model_data.parquet", index=False)

    est = md[md.t <= LAST_DATA_YEAR]
    print(f"entities: {e.opeid6.nunique():,}; entity-years: {len(e):,}")
    print(f"model rows (t<=2023): {len(est):,}; closures in t: {est.closed_t.sum():,}; "
          f"entities that closed 2004-2023: {ent.close_year.between(2004, 2023).sum():,}")
    print(est.groupby("t").closed_t.agg(["size", "sum"]).T.to_string())


if __name__ == "__main__":
    main()
