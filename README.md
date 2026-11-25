# Closure and distress risk: Lewis & Clark and peers

This repo replicates the method in **Kelchen, Ritter & Webber (2024), "Predicting College Closures and Financial Distress," Federal Reserve Bank of Philadelphia Working Paper 24-20** ([PDF](https://www.philadelphiafed.org/-/media/FRBP/Assets/working-papers/2024/wp24-20.pdf)). It then applies the method to Lewis & Clark College and the 33 peers used on the [budget analysis site](https://github.com/drasay-lc/budgetanalysis) (`config/peers.csv`, Groups 1 to 3).

## What it does

1. **Data** (`scripts/fetch_ipeds.sh`, `src/build_panel.py`). Pulls IPEDS complete data files for the 2001-02 to 2023-24 collections:
   * HD: directory and closure dates
   * EF A: fall enrollment
   * EAP: staff
   * F2 / F3: private nonprofit and for-profit finance

   The files come from the public GitHub mirror of NCES's files ([pbradl42/IPEDSpublic](https://github.com/pbradl42/IPEDSpublic)). Run `SOURCE=nces ./scripts/fetch_ipeds.sh` to pull them from NCES directly.
2. **Features** (`src/features.py`). Builds the paper's variables at the 6-digit OPEID level for private Title IV colleges:
   * **Enrollment**: level, change, full-time share, undergraduate share
   * **Staff**: level, change, full-time share, instructional share
   * **Revenue**: level and change, plus tuition, auxiliary, investment, and gifts/grants shares
   * **Expenses**: level and change, plus instruction, scholarship, interest, depreciation, and salary shares
   * **Balance sheet**: unrestricted net assets, debt, endowment
   * **Ratios**: operating margin, days cash, EBIDA, debt/EBIDA, debt/assets, and the changes in these ratios
   * **The paper's four distress flags**:
     * revenue 10% below its 5-year high
     * negative margin in 3 of 5 years
     * enrollment 10% below its 5-year high
     * enrollment down 5% or more three years running

   Dollar values are in FY2023 dollars and skewed levels are logged. Variables are winsorized at 2.5%, and features enter at 2- to 5-year lags.
3. **Models** (`src/modeling.py`, `src/evaluate.py`). Estimates the paper's six model/data pairs A to F (LPM, LASSO-LPM, binned LPM, and three XGBoost variants). Each is trained on 75% of institutions and evaluated on the other 25%. Outcomes are closure in year *t* and closure within *t* to *t*+2.
4. **Scoring** (`src/score.py`). Produces:
   * the current risk (t = 2025, from fall 2023 / FY2023 data) for L&C and every peer
   * cross-fitted risk histories for 2004-2025, so no college is scored by a model trained on itself
   * predicted odds of each distress flag
   * percentiles among the ~1,500 private nonprofit four-year colleges scored in the same year
5. **Scenarios** (`src/scenarios.py`). Runs the paper's enrollment scenarios (Section VI):
   * baseline (flat enrollment)
   * each college's own 2019-23 decline persists
   * a 15% one-time drop in 2025
   * a gradual 15% decline by 2029

   Revenue and expenses scale with enrollment. An extra scenario, not in the paper, keeps half of expenses fixed.

6. **Dashboard data** (`src/export_dashboard.py`). Writes the small CSVs in `dashboard/data/` that feed the results dashboard. The dashboard's page lives in `dashboard/index.html`, `app.js` and `charts.js`, and the out-of-sample check reads `config/later_closures.csv`.

7. **Track record** (`src/closure_backtest.py`, `make backtest`). Asks whether the model would have flagged real closures in advance, using cross-fitted model F scores (no college is scored by a model that saw it). A closure in year *C* counts as flagged if the college was in the riskiest tenth of private nonprofit four-year colleges in any score for *C*-2 to *C*.
   * **Oregon**: every Oregon nonprofit closure since 2006 (`config/oregon_closures.csv`, with sources).
   * **U.S. peer closures, 2016 to 2027**: IPEDS-recorded closures plus a sourced list of later closures and absorptions (`config/closure_events.csv`), screened for resemblance to the comparison set on Carnegie basic class, undergraduate program mix and profile, residential setting, and size (expenses and endowment at least half the smallest in the comparison set, median of the last five fiscal years before closing).
   * Writes `output/backtest_*.csv` and the two-part page in `pages/track-record/` (with `pages/index.html` as a section hub). `make site SITE=../budgetanalysis` copies them into the budget analysis site under `closure-risk/`. Published at https://drasay-lc.github.io/budgetanalysis/closure-risk/ (hub, `lewis-clark-and-peers/`, `track-record/`).

```bash
pip install -r requirements.txt
make all          # about 5 minutes on 2 cores; outputs land in output/
```

## Replication check

AUC on held-out institutions, private colleges, 2004-2023:

| Model | This repo, closed in *t* | Paper (Table 4), closed in *t* | This repo, private nonprofit 4-yr only |
|---|---|---|---|
| A LPM, non-missing | 74.9 | 78.7 | 74.2 |
| B LASSO LPM | 78.6 | 83.4 | 78.5 |
| C LPM, binned | 78.7 | 75.6 | 79.7 |
| D XGBoost, binned | 79.6 | 76.9 | 84.5 |
| E XGBoost, continuous | 82.3 | 80.6 | 86.5 |
| F XGBoost, all controls | 82.9 | 81.8 | 86.8 |
| F, closed within 3 years | 81.9 | 86.8 | 84.8 |

The ranking matches the paper: machine learning beats the linear models, and the all-controls XGBoost is best for closure in *t*.

In model F, the top predicted-risk decile closes within three years at a 30% rate, against 0.06% in the bottom decile. At an 80% true-positive rate, the false-positive rate is 33%, the same as the paper's "about 33% FPR yields roughly 82% TPR."

The most important features are:
* the enrollment-down-10% flag
* endowment
* the revenue-down-10% flag
* enrollment size and change
* expense change

This is consistent with the paper's finding that ratio and change variables beat levels.

**Out-of-sample check.** IPEDS closure data here end in 2023. Among the 1,520 private nonprofit four-year colleges scored for 2025, these later closures rank in the riskiest decile:

* Bacone (3rd)
* University of Valley Forge (25th)
* The King's College (26th)
* Clarks Summit (33rd)
* Fontbonne (107th)
* Northland (136th)
* Birmingham-Southern (140th)

Wells (567th) and Limestone (795th) were missed.

## Results for Lewis & Clark and peers (see `output/`)

* **Current closure risk (t = 2025, model F, within 3 years).**
  * Lewis & Clark: 0.06%, the 11th percentile of private nonprofit four-year colleges (89% of them score as riskier). The median college scores 0.7%.
  * Every peer is below that median. The highest are Saint Mary's CA (0.54%), Drew, Pacific Lutheran, Goucher and Seattle Pacific (0.38% to 0.51%).
* **History.** L&C's cross-fitted percentile:
  * 2005-2008: 33rd to 42nd
  * 2009-2024: 14th to 32nd
  * 2025: 8th
* **Distress outlook for FY2025.** L&C has low predicted odds on all four flags: revenue down 10% (12%), negative margin 3 of 5 (1%), enrollment down 10% (8%), and three-year enrollment slide (0%).
  * Seattle Pacific, Goucher, Capital, Redlands, Hamline and Stetson have high odds on the enrollment flags.
* **Demographic cliff (2031, model F, within 3 years).**
  * With costs scaling, a 15% gradual decline moves L&C from 0.07% to 0.12%.
  * If half of expenses are fixed, it moves to 0.27%. That is the 5th-largest value among the 34 colleges, though still below the private nonprofit four-year median (0.91% in that scenario).
  * L&C is more exposed to fixed costs than most peers because:
    * its FY2023 margin was already slightly negative (-2.9%)
    * tuition is 68% of revenue, tied for the 3rd-highest share in the set
    * it carries $153M of plant debt

## Track record

* **Oregon**: 5 of the 6 testable closures were in the riskiest tenth before closing (OCAC, Concordia, Birthingway, OCOM, Multnomah). Marylhurst peaked at the 87th percentile. Cascade College (reported under Oklahoma Christian) and Oregon College of Ministry are not in the panel.
* **U.S., 2016 to 2027**: 177 private nonprofit four-year closures and absorptions. The model flagged 86 of 131 scored colleges outside the peer screen and 9 of 29 near peers, but none of the 7 peer-sized ones (Wheelock, Mills, Holy Names, Birmingham-Southern, Cabrini, Saint Rose, Hampshire), which peaked between the 19th and 84th percentiles.
* In the riskiest tenth, 11.9% of private nonprofit four-year colleges closed within three years, against 1.3% of the rest (2004 to 2021 scores).

## Differences from the paper (read before citing numbers)

* **Closures** are dated from IPEDS directory files: the year a closure is recorded, or the year after an OPEID stops reporting without a merger record. The paper uses FSA's PEPS closed-school file, which could not be downloaded here. This gives about 1,840 private closures for 2004-2023, against the paper's 1,671 for 1996-2023, with the same 2016-18 peak. Some "stopped reporting" cases are probably losses of Title IV eligibility rather than closures.
* **Federal accountability metrics** (HCM2, FRC) and **county controls** (BEA, SAIPE, LAUS) are not included because their sources were unreachable. The paper finds that county controls add little.
* **IPEDS has no cash line.**
  * *Days cash* is unrestricted net assets divided by daily expenses net of depreciation.
  * *Operating margin* uses total revenue including investment return, so FASB colleges' margins swing with markets.
  * *Debt* is plant-related debt, and debt-to-assets is total liabilities over total assets.
* **The train/test split** is by institution, which is stricter than splitting by observation. XGBoost hyperparameters are fixed, not tuned (see `src/modeling.py`).
* **Scenarios** start from 2023 rather than the paper's 2019. Results are expected closures per year, summed over every private college (`output/scenario_expected_closures.csv`).
* **Latest data** are FY2023 finance and fall 2023 enrollment, the newest year in the mirror. NCES has released 2024-25 data, and pulling it with `SOURCE=nces LAST=2024` would move the scoring year to 2026.

Probabilities are correlational screening scores, not forecasts that a specific college will close.

## Outputs

| File | Contents |
|---|---|
| `model_auc.csv` | AUC by model and outcome |
| `deciles_*.csv` | actual closure share by predicted-risk decile |
| `topn_*.csv` | closures among the N highest-risk institution-years |
| `tpr_fpr_*.csv` | false-positive rate at a target true-positive rate |
| `importance.csv` | XGBoost gain and LPM standardized coefficients |
| `peer_scores_current.csv` | current risk and percentile, models B and F |
| `peer_risk_history.csv` | cross-fitted risk for 2004-2025 |
| `np4_risk_by_year.csv` | median and 90th percentile for private nonprofit four-year colleges |
| `peer_distress_outlook.csv` | predicted odds of each distress flag |
| `peer_metrics.csv` | underlying IPEDS metrics by year |
| `scenario_*.csv` | enrollment-scenario results |
