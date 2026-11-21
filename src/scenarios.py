"""Enrollment scenarios (paper Section VI), applied to every private college and to the peer set.

Each college still reporting in 2023 is projected forward from its 2023 values, in 2023 dollars:
  baseline        enrollment flat at the 2023 level
  persist         each college's own 2019-2023 enrollment decline (if any) continues every year
  cliff_onetime   enrollment drops 15% in 2025 and stays there
  cliff_gradual   enrollment falls in equal steps to 15% below 2023 by 2029
  cliff_gradual_fixed50   (not in the paper) as cliff_gradual, but half of expenses are
                  fixed, so expenses fall only half as fast as enrollment and revenue

As in the paper, revenues, expenses and staff scale with enrollment with constant
shares; balance-sheet items stay at their 2023 values except that unrestricted net
assets absorb any change in the operating result. The paper's approach understates
closures because it ignores fixed costs; the fixed-cost variant shows how much that matters.

Models: LASSO-selected LPM (model B, as in the paper) and XGBoost all-controls (model F).

Writes output/scenario_expected_closures.csv and output/scenario_peers.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import (FEATURES, LAST_DATA_YEAR, LEVEL_COLS, add_features, entity_panel,
                      load_cpi, model_data)
from modeling import fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PROC = ROOT / "data" / "processed"
HORIZON = 2031
SCALE = [c for c in LEVEL_COLS if c.startswith(("enr_", "staff_", "rev_", "exp_"))]
FLOW_REV, FLOW_EXP = "rev_total", "exp_total"


def path(name: str, years, g: pd.Series) -> pd.DataFrame:
    """Enrollment multiplier relative to 2023 for each entity and future year."""
    out = {}
    for y in years:
        k = y - LAST_DATA_YEAR
        if name == "baseline":
            m = pd.Series(1.0, index=g.index)
        elif name == "persist":
            m = (1 + g) ** k
        elif name == "cliff_onetime":
            m = pd.Series(0.85 if y >= 2025 else 1.0, index=g.index)
        else:  # gradual
            step = min(max(y - 2024, 0), 5)
            m = pd.Series(1 - 0.03 * step, index=g.index)
        out[y] = m
    return pd.DataFrame(out)


def project(base: pd.DataFrame, name: str) -> pd.DataFrame:
    last = base[base.year == LAST_DATA_YEAR].set_index("opeid6")
    y19 = base[base.year == 2019].set_index("opeid6")["enr_total"]
    g = ((last["enr_total"] / y19.reindex(last.index)) ** (1 / 4) - 1).clip(-0.15, 0).fillna(0)
    years = list(range(LAST_DATA_YEAR + 1, HORIZON + 1))
    mult = path("gradual" if name.startswith("cliff_gradual") else name, years, g)
    fixed = 0.5 if name.endswith("fixed50") else 0.0
    rows = []
    unres = last["net_assets_unres"].copy()
    base_result = last[FLOW_REV] - last[FLOW_EXP]
    for y in years:
        m = mult[y]
        r = last.copy()
        r["year"] = y
        for c in SCALE:
            f = (fixed + (1 - fixed) * m) if c.startswith("exp_") else m
            r[c] = last[c] * f
        result = r[FLOW_REV] - r[FLOW_EXP]
        unres = unres + (result - base_result).fillna(0)
        r["net_assets_unres"] = unres
        rows.append(r.reset_index())
    fut = pd.concat(rows, ignore_index=True)
    return pd.concat([base, fut], ignore_index=True)


def main() -> None:
    base, _ = entity_panel()
    active = set(base.loc[base.year == LAST_DATA_YEAR, "opeid6"])
    md = pd.read_parquet(PROC / "model_data.parquet")
    lagged = [c for c in md.columns if "_L" in c and c.rsplit("_L", 1)[0] in FEATURES]
    lo, hi = md[lagged].min(), md[lagged].max()       # winsorization bounds used in training
    models = {
        ("closed_t", "B"): fit("B", md[md.t <= LAST_DATA_YEAR], "closed_t"),
        ("closed_t", "F"): fit("F", md[md.t <= LAST_DATA_YEAR], "closed_t"),
        ("closed_3yr", "F"): fit("F", md[md.t <= LAST_DATA_YEAR - 2], "closed_3yr"),
    }
    cpi = load_cpi()
    for y in range(LAST_DATA_YEAR + 1, HORIZON + 1):
        cpi.loc[y] = 1.0  # projections are already in 2023 dollars

    peers = pd.read_csv(ROOT / "config" / "peers.csv")
    unit2ope = base.drop_duplicates("UNITID").set_index("UNITID")["opeid6"]
    peers["opeid6"] = peers.unitid.map(unit2ope)

    agg, peer_rows = [], []
    for name in ["baseline", "persist", "cliff_onetime", "cliff_gradual", "cliff_gradual_fixed50"]:
        e = add_features(project(base, name), cpi=cpi)
        d = model_data(e, t_last=HORIZON + 2)
        d = d[(d.t >= LAST_DATA_YEAR + 3) & d.opeid6.isin(active)].copy()
        d[lagged] = d[lagged].clip(lo, hi, axis=1)
        for (y, k), m in models.items():
            d[f"p_{y}_{k}"] = np.clip(m.predict(d), 0, 1)
        a = d.groupby("t")[["p_closed_t_B", "p_closed_t_F"]].sum()
        np4 = d[(d.forprofit == 0) & (d.four_year == 1)]
        a = a.join(np4.groupby("t")[["p_closed_t_B", "p_closed_t_F"]].sum().add_suffix("_np4"))
        a["scenario"] = name
        agg.append(a.reset_index())
        pr = d[d.opeid6.isin(set(peers.opeid6))][["opeid6", "t", "p_closed_3yr_F", "p_closed_t_B"]]
        ref = np4.groupby("t")["p_closed_3yr_F"]
        pr = pr.merge(ref.median().rename("np4_median_3yr_F"), on="t")
        pr["scenario"] = name
        peer_rows.append(pr)
        print(name, a.round(1).to_string())

    out = pd.concat(agg)
    basel = out[out.scenario == "baseline"].set_index("t")
    for c in ["p_closed_t_B", "p_closed_t_F", "p_closed_t_B_np4", "p_closed_t_F_np4"]:
        out[c + "_vs_baseline"] = out[c] - out.t.map(basel[c])
    out.to_csv(OUT / "scenario_expected_closures.csv", index=False)
    pr = pd.concat(peer_rows).merge(peers[["opeid6", "key", "short", "peer_group"]], on="opeid6")
    pr.to_csv(OUT / "scenario_peers.csv", index=False)


if __name__ == "__main__":
    main()
