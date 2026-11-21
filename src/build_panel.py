"""Assemble an institution-year IPEDS panel (2002-2023) for private colleges.

One row per UNITID x IPEDS collection year t, where
  * directory fields come from HD{t}
  * enrollment is fall t (EF{t}A)
  * staff is fall t (EAP{t})
  * finance is fiscal year t (July t-1 to June t; F{t-1}{t}_F2 or _F3)

Closure dates come from the HD files (DEATHYR / CLOSEDAT / NEWID), which NCES
populates when an institution leaves the IPEDS universe. Institutions that
left by merging into another UNITID (NEWID > 0) are not counted as closures.

Output: data/processed/panel.parquet
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
YEARS = range(2001, 2024)


def _read(path: Path, usecols=None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = [c.strip().strip('"').upper() for c in df.columns]
    if usecols is not None:
        df = df[[c for c in usecols if c in df.columns]]
    return df


def _pick(folder: Path, stem: str) -> Path | None:
    """Prefer NCES revised files (_rv) when present."""
    rv = folder / f"{stem}_rv.csv"
    base = folder / f"{stem}.csv"
    return rv if rv.exists() else (base if base.exists() else None)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------- directory
def read_hd(y: int) -> pd.DataFrame | None:
    p = _pick(RAW / str(y), f"hd{y}")
    if p is None:
        return None
    cols = ["UNITID", "INSTNM", "STABBR", "OPEID", "SECTOR", "ICLEVEL", "CONTROL",
            "DEGGRANT", "HLOFFER", "NEWID", "DEATHYR", "CLOSEDAT", "CYACTIVE",
            "COUNTYCD", "FIPS"]
    d = _read(p, cols)
    d["year"] = y
    return d


# --------------------------------------------------------------- enrollment
def read_ef(y: int) -> pd.DataFrame | None:
    p = _pick(RAW / str(y), f"ef{y}a")
    if p is None:
        return None
    d = _read(p)
    if "EFTOTLT" in d.columns:
        tot = "EFTOTLT"
    elif "EFRACE24" in d.columns:
        tot = "EFRACE24"
    else:  # 2001: grand total = total men (15) + total women (16)
        d["EFRACE24"] = _num(d["EFRACE15"]) + _num(d["EFRACE16"])
        tot = "EFRACE24"
    if "EFALEVEL" not in d.columns:  # 2001 layout: map survey line numbers to levels
        line = _num(d["LINE"])
        d["EFALEVEL"] = np.select([line == 29, line == 14, line.isin([8, 22])], [1, 21, 2], -1)
    d = d[["UNITID", "EFALEVEL", tot]].rename(columns={tot: "n"})
    d["n"] = _num(d["n"])
    w = d.pivot_table(index="UNITID", columns="EFALEVEL", values="n", aggfunc="sum")
    out = pd.DataFrame({
        "enr_total": w.get(1),
        "enr_ug": w.get(2),
        "enr_ft": w.get(21),
    }).reset_index()
    out["year"] = y
    return out


# -------------------------------------------------------------------- staff
def read_eap(y: int) -> pd.DataFrame | None:
    p = _pick(RAW / str(y), f"eap{y}")
    if p is None:
        return None
    d = _read(p)
    if "EAPTOT" not in d.columns:  # 2001 used a different layout; staff starts in 2002
        return None
    d["EAPTOT"] = _num(d["EAPTOT"])
    if "EAPRECTP" in d.columns:  # 2001-2011 layout
        code = _num(d["EAPRECTP"])
        g = lambda c: d.loc[code == c].groupby("UNITID")["EAPTOT"].sum()
        out = pd.DataFrame({
            "staff_total": g(1100),
            "staff_ft": g(2100),
            # primarily instruction + instruction combined with research/public service
            "staff_instr": g(1110).add(g(1120), fill_value=0),
        })
    else:  # 2012+ layout
        cat = _num(d["EAPCAT"])
        d["EAPFT"] = _num(d.get("EAPFT"))
        out = pd.DataFrame({
            "staff_total": d.loc[cat == 10000].groupby("UNITID")["EAPTOT"].sum(),
            "staff_ft": d.loc[cat == 10000].groupby("UNITID")["EAPFT"].sum(),
            "staff_instr": d.loc[cat == 21000].groupby("UNITID")["EAPTOT"].sum(),
        })
    out = out.reset_index()
    out["year"] = y
    return out


# ------------------------------------------------------------------ finance
def _ftag(y: int) -> str:
    return f"f{(y - 1) % 100:02d}{y % 100:02d}"


def read_f2(y: int) -> pd.DataFrame | None:
    """Private nonprofit (FASB) finance, fiscal year y."""
    p = _pick(RAW / str(y), f"{_ftag(y)}_f2")
    if p is None:
        return None
    d = _read(p)
    g = lambda c: _num(d[c]) if c in d.columns else np.nan
    out = pd.DataFrame({
        "UNITID": d["UNITID"],
        "rev_total": g("F2D16").fillna(g("F2B01")) if "F2D16" in d else g("F2B01"),
        "rev_tuition": g("F2D01"),
        "rev_aux": g("F2D12"),
        "rev_invest": g("F2D10"),
        "rev_gifts_grants": pd.concat([g(c) for c in ["F2D05", "F2D06", "F2D07", "F2D08", "F2D09"]
                                       if c in d.columns], axis=1).sum(axis=1, min_count=1),
        "exp_total": g("F2B02"),
        "exp_instruction": g("F2E011"),
        "exp_scholarships": g("F2E081"),
        "exp_salaries": g("F2E132"),
        "exp_depreciation": g("F2E135"),
        "exp_interest": g("F2E136"),
        "assets_total": g("F2A02"),
        "liab_total": g("F2A03"),
        "debt_ppe": g("F2A03A"),
        "net_assets_unres": g("F2A04"),
        "net_assets_temp": g("F2A05B"),
        "ppe_net": g("F2A19"),
        "endowment": g("F2H02"),
    })
    out["fin_form"] = "F2"
    out["year"] = y
    return out


def read_f3(y: int) -> pd.DataFrame | None:
    """Private for-profit finance, fiscal year y (layout changed in FY2014)."""
    p = _pick(RAW / str(y), f"{_ftag(y)}_f3")
    if p is None:
        return None
    d = _read(p)
    g = lambda c: _num(d[c]) if c in d.columns else pd.Series(np.nan, index=d.index)
    new = "F3E011" in d.columns
    out = pd.DataFrame({
        "UNITID": d["UNITID"],
        "rev_total": g("F3D09").fillna(g("F3B01")),
        "rev_tuition": g("F3D01"),
        "rev_aux": g("F3D07"),
        "rev_invest": g("F3D05"),
        "rev_gifts_grants": pd.concat([g("F3D02"), g("F3D03"), g("F3D04")], axis=1).sum(axis=1, min_count=1),
        "exp_total": g("F3B02"),
        "exp_instruction": g("F3E011") if new else g("F3E01"),
        "exp_scholarships": g("F3E051") if new else g("F3E05"),
        "exp_salaries": g("F3E072"),
        "exp_depreciation": g("F3E075"),
        "exp_interest": g("F3E076"),
        "assets_total": g("F3A01"),
        "liab_total": g("F3A02"),
        "debt_ppe": g("F3A02A"),
        "net_assets_unres": g("F3A03"),  # total equity
        "net_assets_temp": np.nan,
        "ppe_net": g("F3A01B"),
        "endowment": np.nan,
    })
    out["fin_form"] = "F3"
    out["year"] = y
    return out


# ------------------------------------------------------------------ closures
def _parse_closedat(s: pd.Series) -> pd.Series:
    def one(v):
        v = str(v).strip()
        m = re.search(r"(\d{4})$", v) or re.search(r"^(\d{4})", v)
        if m:
            return int(m.group(1))
        m = re.search(r"[-/](\d{2})$", v)  # e.g. 06/30/09
        return 2000 + int(m.group(1)) if m else np.nan
    return s.map(one)


def closure_table(hd: pd.DataFrame) -> pd.DataFrame:
    """One row per UNITID with closure year (NaN if still open or merged)."""
    hd = hd.sort_values(["UNITID", "year"])
    last = hd.groupby("UNITID").tail(1).copy()
    first_year = hd.groupby("UNITID")["year"].min().rename("first_year")
    last_year = hd.groupby("UNITID")["year"].max().rename("last_year")
    last = last.join(first_year, on="UNITID").join(last_year, on="UNITID")

    newid = _num(last["NEWID"]).fillna(-2)
    deathyr = _num(last["DEATHYR"]).fillna(-2)
    cyactive = _num(last["CYACTIVE"]).fillna(1)
    cyear = _parse_closedat(last["CLOSEDAT"].astype(str).where(last["CLOSEDAT"].astype(str) != "-2", ""))

    flagged = (deathyr > 0) | cyear.notna() | (cyactive != 1)
    merged = newid > 0
    max_year = hd["year"].max()
    # vanished before the last HD file without a merger record -> treat as closed after last appearance
    vanished = (last["last_year"] < max_year) & ~merged

    close_year = np.where(cyear.notna(), cyear, np.where(deathyr > 0, deathyr, np.nan))
    close_year = np.where(np.isnan(close_year) & vanished, last["last_year"] + 1, close_year)
    close_year = np.where(np.isnan(close_year) & flagged & ~merged, last["last_year"], close_year)
    close_year = np.where(merged, np.nan, close_year)

    return pd.DataFrame({
        "UNITID": last["UNITID"].values,
        "close_year": close_year,
        "merged": merged.values,
        "first_year": last["first_year"].values,
        "last_year": last["last_year"].values,
    })


def main() -> None:
    hd = pd.concat([d for y in YEARS if (d := read_hd(y)) is not None], ignore_index=True)
    ef = pd.concat([d for y in YEARS if (d := read_ef(y)) is not None], ignore_index=True)
    eap = pd.concat([d for y in YEARS if (d := read_eap(y)) is not None], ignore_index=True)
    f2 = pd.concat([d for y in YEARS if (d := read_f2(y)) is not None], ignore_index=True)
    f3 = pd.concat([d for y in YEARS if (d := read_f3(y)) is not None], ignore_index=True)
    fin = pd.concat([f2, f3], ignore_index=True).drop_duplicates(["UNITID", "year"], keep="first")

    for c in ["CONTROL", "SECTOR", "ICLEVEL", "DEGGRANT", "HLOFFER"]:
        hd[c] = _num(hd[c])
    clos = closure_table(hd)

    # skeleton: every UNITID x year it appears in HD, plus 2001 rows for lagging (no HD2001)
    base = hd[["UNITID", "year", "INSTNM", "STABBR", "OPEID", "CONTROL", "SECTOR", "ICLEVEL",
               "DEGGRANT", "HLOFFER"]]
    first2002 = base[base.year == 2002].copy()
    first2002["year"] = 2001
    base = pd.concat([first2002, base], ignore_index=True)

    panel = (base.merge(ef, on=["UNITID", "year"], how="left")
                 .merge(eap, on=["UNITID", "year"], how="left")
                 .merge(fin, on=["UNITID", "year"], how="left")
                 .merge(clos, on="UNITID", how="left"))
    panel["OPEID"] = _num(panel["OPEID"]).astype("Int64").astype(str).str.zfill(8)
    panel["INSTNM"] = panel["INSTNM"].astype(str)
    panel["STABBR"] = panel["STABBR"].astype(str)
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT / "panel.parquet", index=False)
    clos.to_csv(OUT / "closures.csv", index=False)

    priv = panel[panel.CONTROL.isin([2, 3]) & (panel.year >= 2002)]
    cl = clos.merge(priv.groupby("UNITID")["CONTROL"].first(), on="UNITID")
    print(f"panel rows: {len(panel):,}; private rows: {len(priv):,}; private institutions: {priv.UNITID.nunique():,}")
    print("private closures by year:\n", cl.dropna(subset=['close_year']).groupby('close_year').size().to_string())


if __name__ == "__main__":
    main()
