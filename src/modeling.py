"""Model definitions shared by evaluate.py, score.py and scenarios.py.

Six model/data pairs, as in Kelchen, Ritter & Webber (2024), Table 4:
  A  LPM (OLS), continuous variables, non-missing sample
  B  LASSO-selected LPM, continuous variables (LASSO picks covariates on the
     non-missing sample; OLS is refit on every row where the picked covariates exist)
  C  LPM, binned variables (quintiles + a missing bin), full sample
  D  XGBoost, binned variables, full sample
  E  XGBoost, continuous variables (missing values handled natively), full sample
  F  XGBoost, all controls: continuous variables at lags 2-5, full sample
A-E use two- and three-year lags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from features import FEATURES, STATIC

SEED = 20241231
BASE_LAGS = (2, 3)
ALL_LAGS = (2, 3, 4, 5)
MODELS = {
    "A": "LPM, continuous, non-missing sample",
    "B": "LASSO-selected LPM, continuous",
    "C": "LPM, binned variables, full sample",
    "D": "XGBoost, binned variables, full sample",
    "E": "XGBoost, continuous variables, full sample",
    "F": "XGBoost, all controls (lags 2-5), full sample",
}
XGB_PARAMS = dict(n_estimators=500, max_depth=4, learning_rate=0.04, subsample=0.8,
                  colsample_bytree=0.7, min_child_weight=5, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="auc", tree_method="hist",
                  n_jobs=2, random_state=SEED)


def cols(lags) -> list[str]:
    return [f"{f}_L{l}" for l in lags for f in FEATURES] + STATIC


# ------------------------------------------------------------- binning
class Binner:
    """Quintile bins learned on training data, plus a 'missing' bin for every variable."""

    def fit(self, X: pd.DataFrame):
        self.edges = {}
        for c in X.columns:
            s = X[c].dropna()
            if s.nunique() <= 2:
                self.edges[c] = None
            else:
                q = np.unique(s.quantile([0.2, 0.4, 0.6, 0.8]).values)
                self.edges[c] = q
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in X.columns:
            s = X[c]
            e = self.edges[c]
            if e is None:
                out[f"{c}=1"] = (s == 1).astype(float)
            else:
                b = np.digitize(s.fillna(np.inf), e)  # 0..len(e)
                for k in range(1, len(e) + 1):
                    out[f"{c}=q{k}"] = ((b == k) & s.notna()).astype(float)
            out[f"{c}=NA"] = s.isna().astype(float)
        return pd.DataFrame(out, index=X.index)


# ------------------------------------------------------------- fitted model wrapper
class Fitted:
    def __init__(self, kind, model, columns, binner=None, selected=None):
        self.kind, self.model, self.columns, self.binner, self.selected = kind, model, columns, binner, selected

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.columns]
        if self.kind in ("A", "B"):
            use = self.selected if self.selected is not None else self.columns
            Xs = sm.add_constant(X[use], has_constant="add")
            p = np.array(self.model.predict(Xs), dtype=float)
            p[X[use].isna().any(axis=1).values] = np.nan  # undefined where inputs are missing
            return p
        if self.kind == "C":
            Xb = sm.add_constant(self.binner.transform(X), has_constant="add")
            return np.asarray(self.model.predict(Xb[self.model.model.exog_names]), dtype=float)
        if self.kind == "D":
            return self.model.predict_proba(self.binner.transform(X))[:, 1]
        return self.model.predict_proba(X)[:, 1]


def fit(kind: str, train: pd.DataFrame, y: str) -> Fitted:
    if kind in ("A", "B", "C", "D", "E"):
        c = cols(BASE_LAGS)
    else:
        c = cols(ALL_LAGS)
    X, Y = train[c], train[y]
    if kind == "A":
        ok = X.notna().all(axis=1)
        Xa = X[ok].loc[:, X[ok].std() > 0]
        m = sm.OLS(Y[ok], sm.add_constant(Xa)).fit()
        return Fitted("A", m, c, selected=list(Xa.columns))
    if kind == "B":
        ok = X.notna().all(axis=1)
        Xa = X[ok].loc[:, X[ok].std() > 0]
        sc = StandardScaler().fit(Xa)
        lasso = LassoCV(cv=5, random_state=SEED, alphas=60, max_iter=5000).fit(sc.transform(Xa), Y[ok])
        sel = [col for col, b in zip(Xa.columns, lasso.coef_) if abs(b) > 1e-10]
        if not sel:
            sel = list(Xa.columns[:1])
        ok2 = X[sel].notna().all(axis=1)
        m = sm.OLS(Y[ok2], sm.add_constant(X.loc[ok2, sel])).fit()
        return Fitted("B", m, c, selected=sel)
    if kind in ("C", "D"):
        b = Binner().fit(X)
        Xb = b.transform(X)
        Xb = Xb.loc[:, Xb.std() > 0]
        if kind == "C":
            m = sm.OLS(Y, sm.add_constant(Xb)).fit()
        else:
            m = xgb.XGBClassifier(**XGB_PARAMS).fit(Xb, Y)
            b.out_cols = list(Xb.columns)
            orig = b.transform
            b.transform = lambda Z, _o=orig, _c=list(Xb.columns): _o(Z)[_c]
        if kind == "C":
            orig = b.transform
            b.transform = lambda Z, _o=orig, _c=list(Xb.columns): _o(Z)[_c]
        return Fitted(kind, m, c, binner=b)
    m = xgb.XGBClassifier(**XGB_PARAMS).fit(X, Y)
    return Fitted(kind, m, c)


def base_name(col: str) -> str:
    return col.split("=")[0].rsplit("_L", 1)[0]
