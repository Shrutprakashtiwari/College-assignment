# CSET485 – AI and Society | Adult Census Income Assignment
# Random state: 526 | Train-test split: 70:30
# Run in Google Colab or Jupyter. Internet is required to download the UCI dataset.

# If needed, uncomment:
# %pip install -q pandas numpy scipy scikit-learn statsmodels matplotlib

import io
import zipfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

warnings.filterwarnings("ignore")
SEED = 526
OUT = Path("adult_assignment_outputs")
OUT.mkdir(exist_ok=True)

# -------------------------
# 1. Load Adult dataset
# -------------------------
# UCI Adult files: adult.data (train source) and adult.test (test source).
# We combine the source files, then make the assignment's own 70:30 split.
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]

def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8", errors="replace")

try:
    train_text = fetch_text(BASE + "adult.data")
    test_text = fetch_text(BASE + "adult.test")
    # adult.test begins with a comment line; remove it.
    test_lines = [line for line in test_text.splitlines() if line.strip() and not line.lstrip().startswith("|")]
    train_lines = [line for line in train_text.splitlines() if line.strip() and not line.lstrip().startswith("|")]
    combined_text = "\n".join(train_lines + test_lines)
    df = pd.read_csv(
        io.StringIO(combined_text),
        header=None, names=COLS, skipinitialspace=True, na_values=["?", " ?"]
    )
    print("Loaded UCI Adult dataset directly from UCI.")
except Exception as e:
    raise RuntimeError(
        "Could not download Adult dataset from UCI. Upload adult.data and adult.test "
        "to Colab/Jupyter, or download them manually from "
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/ "
        f"and adjust the loading cell. Original error: {e}"
    )

# Clean whitespace and labels. adult.test labels can end in a period.
for c in df.select_dtypes(include="object").columns:
    # Keep ordinary object dtype with np.nan; sklearn SimpleImputer can fail on pandas StringDtype/pd.NA.
    df[c] = df[c].str.strip().astype(object)
df["income"] = df["income"].astype(object).str.replace(".", "", regex=False).str.strip()
df = df[df["income"].isin(["<=50K", ">50K"])].copy()
df["income_binary"] = (df["income"] == ">50K").astype(int)

print("\nDATASET OVERVIEW")
print("Rows:", len(df), "| columns including target:", len(df.columns))
print("Target counts:\n", df["income"].value_counts(dropna=False))
print("\nMissing values before imputation:\n", df[COLS].isna().sum().sort_values(ascending=False))
print("\nDuplicate rows:", int(df.duplicated().sum()))

# -------------------------
# 2. Preprocessing and split
# -------------------------
# Keep demographic variables for group audits; they are not automatically excluded
# from the predictive model. For a fairness audit, report this design choice.
TARGET = "income_binary"
FEATURES = [c for c in COLS if c != "income"]
X = df[FEATURES].copy()
y = df[TARGET].copy()

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
print("\nSPLIT")
print("Train rows:", len(X_train), "| test rows:", len(X_test))
print("Train target rate:", round(float(y_train.mean()), 4))
print("Test target rate:", round(float(y_test.mean()), 4))
print("Random state:", SEED, "| test size:", "30%")

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()
# statsmodels needs dense arrays; Adult data is manageable for this assignment.
X_train_arr = X_train_t.toarray() if hasattr(X_train_t, "toarray") else np.asarray(X_train_t)
X_test_arr = X_test_t.toarray() if hasattr(X_test_t, "toarray") else np.asarray(X_test_t)

# -------------------------
# 3. Part 1: OLS and logistic regression
# -------------------------
ols = LinearRegression()
ols.fit(X_train_arr, y_train)
ols_pred = ols.predict(X_test_arr)

logit = LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED)
logit.fit(X_train_arr, y_train)
logit_prob = logit.predict_proba(X_test_arr)[:, 1]
logit_label = (logit_prob >= 0.50).astype(int)

median_ols = float(np.median(ols_pred))
ols_label = (ols_pred >= median_ols).astype(int)
cm = confusion_matrix(ols_label, logit_label, labels=[0, 1])
mismatch = int(np.sum(ols_label != logit_label))
mismatch_pct = mismatch / len(y_test) * 100

print("\nPART 1: CLASSIFICATION AGREEMENT")
print("Median OLS prediction:", median_ols)
print("Confusion matrix (rows=OLS labels [0,1], columns=Logistic labels [0,1]):\n", cm)
print("Mismatch count:", mismatch)
print("Mismatch percentage:", round(mismatch_pct, 3), "%")
print("OLS median-threshold positive count:", int(ols_label.sum()))
print("Logistic 0.5-threshold positive count:", int(logit_label.sum()))
print("\nLogistic test metrics against actual income:")
print("Accuracy:", round(accuracy_score(y_test, logit_label), 4))
print("Precision:", round(precision_score(y_test, logit_label, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, logit_label, zero_division=0), 4))
print("F1:", round(f1_score(y_test, logit_label, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, logit_prob), 4))

# -------------------------
# 4. Part 2.1: subgroup coefficient comparison
# -------------------------
# Fit statsmodels Logit on processed design matrix for comparable coefficients.
# Use AGE as key predictor (scaled numerical feature) and SEX as subgroup.
# We use a manageable subset of encoded columns for statistical inference to avoid
# very large, slow models: all numerical variables + selected categorical controls.
# This inference model uses raw predictors with formula-based categorical encoding.

import statsmodels.formula.api as smf

analysis = df.copy()
analysis["income_binary"] = analysis["income_binary"].astype(int)
# Simple explicit formula. Categorical fields are handled by statsmodels.
formula_full = (
    "income_binary ~ age + education_num + hours_per_week + capital_gain + capital_loss "
    "+ C(sex) + C(race) + C(marital_status) + C(workclass) + C(occupation)"
)
def fit_formula(frame, formula):
    model = smf.logit(formula=formula, data=frame).fit(disp=False, maxiter=300)
    return model

def safe_fit_formula(frame, formula, label):
    try:
        return fit_formula(frame, formula)
    except Exception as e:
        print(f"Could not fit {label}: {e}")
        return None

full_inf = safe_fit_formula(analysis, formula_full, "full subgroup-inference model")
subgroup_rows = []
key = "education_num"
if full_inf is not None:
    for group_name, group_df in [
        ("Full dataset", analysis),
        ("Sex = Female", analysis[analysis["sex"] == "Female"]),
        ("Sex = Male", analysis[analysis["sex"] == "Male"])
    ]:
        m = full_inf if group_name == "Full dataset" else safe_fit_formula(
            group_df, formula_full, group_name
        )
        if m is not None and key in m.params.index:
            ci = m.conf_int().loc[key]
            subgroup_rows.append({
                "group": group_name, "predictor": key,
                "coef": float(m.params[key]),
                "ci_lower": float(ci.iloc[0]),
                "ci_upper": float(ci.iloc[1]),
                "p_value": float(m.pvalues[key]),
                "n": int(m.nobs)
            })
subgroup_df = pd.DataFrame(subgroup_rows)
print("\nPART 2.1: SUBGROUP COEFFICIENTS (education_num)")
print(subgroup_df.to_string(index=False) if len(subgroup_df) else "No coefficient table available.")
subgroup_df.to_csv(OUT / "subgroup_coefficients.csv", index=False)

# Also search for sign reversal or notable differences across selected predictors.
if len(subgroup_df) == 3:
    print("Coefficient sign pattern:", subgroup_df[["group", "coef"]].to_dict("records"))
    print("Note: overlapping/non-overlapping CIs are descriptive; a formal interaction test is stronger.")
# Interaction test: sex × education_num in full data.
formula_interaction = formula_full + " + education_num:C(sex)"
interaction_model = safe_fit_formula(analysis, formula_interaction, "sex interaction model")
if interaction_model is not None:
    print("\nSex-by-education interaction terms:")
    print(interaction_model.params[interaction_model.params.index.str.contains("education_num:C\\(sex\\)")])
    print(interaction_model.pvalues[interaction_model.pvalues.index.str.contains("education_num:C\\(sex\\)")])

# -------------------------
# 5. Part 2.2: omitted-variable analysis
# -------------------------
# Drop education_num from a formula-based Logistic Regression and compare four predictors.
# This is an omitted-variable sensitivity experiment; coefficient changes in logit
# models are not exclusively attributable to classical confounding.
reduced_formula = (
    "income_binary ~ age + hours_per_week + capital_gain + capital_loss "
    "+ C(sex) + C(race) + C(marital_status) + C(workclass) + C(occupation)"
)
full_omit = safe_fit_formula(analysis, formula_full, "full omitted-variable model")
reduced_omit = safe_fit_formula(analysis, reduced_formula, "reduced omitted-variable model")
omit_rows = []
compare_terms = ["age", "hours_per_week", "capital_gain", "capital_loss"]
if full_omit is not None and reduced_omit is not None:
    for term in compare_terms:
        a = float(full_omit.params.get(term, np.nan))
        b = float(reduced_omit.params.get(term, np.nan))
        pct = ((b - a) / abs(a) * 100) if np.isfinite(a) and abs(a) > 1e-10 else np.nan
        omit_rows.append({
            "predictor": term, "full_coefficient": a,
            "reduced_coefficient": b, "signed_pct_change": pct,
            "absolute_pct_change": abs(pct) if np.isfinite(pct) else np.nan
        })
omit_df = pd.DataFrame(omit_rows)
print("\nPART 2.2: OMITTED-VARIABLE COEFFICIENT CHANGES")
print("Dropped variable: education_num")
print(omit_df.to_string(index=False) if len(omit_df) else "No coefficient table available.")
omit_df.to_csv(OUT / "omitted_variable_changes.csv", index=False)

# -------------------------
# 6. Part 2.3: high-confidence wrong predictions
# -------------------------
test_results = X_test.reset_index(drop=True).copy()
test_results["actual"] = np.asarray(y_test).astype(int)
test_results["probability"] = logit_prob
test_results["prediction_05"] = logit_label
test_results["error_type"] = ""
fp_mask = (test_results["probability"] >= 0.75) & (test_results["actual"] == 0)
fn_mask = (test_results["probability"] <= 0.25) & (test_results["actual"] == 1)
test_results.loc[fp_mask, "error_type"] = "False positive"
test_results.loc[fn_mask, "error_type"] = "False negative"
wrong = test_results[test_results["error_type"] != ""].copy()
wrong_total = len(wrong)
wrong_pct = wrong_total / len(test_results) * 100

print("\nPART 2.3: ADVERSARIAL SUBSET")
print("High-confidence wrong predictions:", wrong_total)
print("Percentage of test set:", round(wrong_pct, 3), "%")
print("False positives:", int(fp_mask.sum()))
print("False negatives:", int(fn_mask.sum()))
if wrong_total > 50:
    wrong_sample = wrong.sample(n=50, random_state=SEED)
else:
    wrong_sample = wrong.copy()
print("Rows in reported subset:", len(wrong_sample))
if len(wrong_sample):
    print("\nSubset age summary:\n", wrong_sample["age"].describe())
    print("\nFull test age summary:\n", test_results["age"].describe())
    for col in ["sex", "race", "marital_status", "education", "occupation"]:
        if col in wrong_sample.columns:
            sub_pct = wrong_sample[col].value_counts(normalize=True, dropna=False).mul(100).round(2)
            full_pct = test_results[col].value_counts(normalize=True, dropna=False).mul(100).round(2)
            compare = pd.concat([sub_pct.rename("subset_pct"), full_pct.rename("full_test_pct")], axis=1).fillna(0)
            print(f"\n{col} distribution (%):\n", compare.to_string())
wrong_sample.to_csv(OUT / "adversarial_subset.csv", index=False)

# -------------------------
# 7. Part 2.4: fairness threshold sweep
# -------------------------
# Sensitive attribute: sex; groups Female and Male.
# Ratios are A/B where A=Female, B=Male. Undefined when denominator is zero.
def safe_ratio(a, b):
    if b is None or not np.isfinite(b) or abs(b) < 1e-12:
        return np.nan
    return a / b

y_test_arr = np.asarray(y_test).astype(int)
sex_test = X_test["sex"].astype("string").reset_index(drop=True)
fair_rows = []
thresholds = np.round(np.arange(0.05, 0.951, 0.10), 2)

for t in thresholds:
    pred = (logit_prob >= t).astype(int)
    metrics = {}
    for group in ["Female", "Male"]:
        mask = (sex_test == group).to_numpy()
        yg, pg = y_test_arr[mask], pred[mask]
        pos_rate = float(pg.mean()) if len(pg) else np.nan
        positives = (pg == 1)
        ppv = float(yg[positives].mean()) if positives.any() else np.nan
        actual_pos = (yg == 1)
        actual_neg = (yg == 0)
        tpr = float(((pg == 1) & actual_pos).sum() / actual_pos.sum()) if actual_pos.sum() else np.nan
        fpr = float(((pg == 1) & actual_neg).sum() / actual_neg.sum()) if actual_neg.sum() else np.nan
        metrics[group] = {"pos_rate": pos_rate, "ppv": ppv, "tpr": tpr, "fpr": fpr}
    A, B = metrics["Female"], metrics["Male"]
    dp = safe_ratio(A["pos_rate"], B["pos_rate"])
    tpr_ratio = safe_ratio(A["tpr"], B["tpr"])
    fpr_ratio = safe_ratio(A["fpr"], B["fpr"])
    eo = np.nanmin([tpr_ratio, fpr_ratio]) if not (np.isnan(tpr_ratio) and np.isnan(fpr_ratio)) else np.nan
    pp = safe_ratio(A["ppv"], B["ppv"])
    fair_rows.append({
        "threshold": float(t), "DP_ratio_Female_over_Male": dp,
        "EO_min_TPR_FPR_ratio": eo, "PP_ratio_Female_over_Male": pp,
        "Female_positive_rate": A["pos_rate"], "Male_positive_rate": B["pos_rate"],
        "Female_TPR": A["tpr"], "Male_TPR": B["tpr"],
        "Female_FPR": A["fpr"], "Male_FPR": B["fpr"],
        "Female_PPV": A["ppv"], "Male_PPV": B["ppv"],
        "accuracy": accuracy_score(y_test_arr, pred)
    })

fair_df = pd.DataFrame(fair_rows)
print("\nPART 2.4: FAIRNESS THRESHOLD SWEEP")
print(fair_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
fair_df.to_csv(OUT / "fairness_thresholds.csv", index=False)

plt.figure(figsize=(8, 6))
valid = fair_df.dropna(subset=["DP_ratio_Female_over_Male", "EO_min_TPR_FPR_ratio"])
plt.scatter(valid["DP_ratio_Female_over_Male"], valid["EO_min_TPR_FPR_ratio"])
for _, r in valid.iterrows():
    plt.annotate(f'{r["threshold"]:.2f}', (r["DP_ratio_Female_over_Male"], r["EO_min_TPR_FPR_ratio"]),
                 xytext=(4, 4), textcoords="offset points", fontsize=8)
plt.axvline(1.0, linestyle="--", linewidth=1)
plt.axhline(1.0, linestyle="--", linewidth=1)
plt.xlim(0, 1.5)
plt.ylim(0, 1.5)
plt.xlabel("Demographic Parity ratio (Female / Male)")
plt.ylabel("Equalized Odds summary ratio")
plt.title("Fairness trade-off by decision threshold")
plt.grid(True, alpha=0.25)
plt.tight_layout()
chart_path = OUT / "dp_vs_eo_tradeoff.png"
plt.savefig(chart_path, dpi=180)
plt.show()
print("Saved fairness chart:", chart_path.resolve())

# -------------------------
# 8. Trade-off calculation helper
# -------------------------
# Choose two rows after inspecting the table, then set t1 and t2 below.
def compare_thresholds(t1, t2):
    r1 = fair_df.loc[np.isclose(fair_df["threshold"], t1)].iloc[0]
    r2 = fair_df.loc[np.isclose(fair_df["threshold"], t2)].iloc[0]
    dp1, dp2 = r1["DP_ratio_Female_over_Male"], r2["DP_ratio_Female_over_Male"]
    eo1, eo2 = r1["EO_min_TPR_FPR_ratio"], r2["EO_min_TPR_FPR_ratio"]
    print(f"Threshold {t1:.2f}: DP={dp1:.4f}, EO={eo1:.4f}, PP={r1['PP_ratio_Female_over_Male']:.4f}")
    print(f"Threshold {t2:.2f}: DP={dp2:.4f}, EO={eo2:.4f}, PP={r2['PP_ratio_Female_over_Male']:.4f}")
    print("DP change:", dp2 - dp1, "| EO change:", eo2 - eo1)
    if np.isfinite(dp1) and np.isfinite(dp2) and abs(dp1) > 1e-12:
        print("Relative DP change (%):", (dp2-dp1)/abs(dp1)*100)
    if np.isfinite(eo1) and np.isfinite(eo2) and abs(eo1) > 1e-12:
        print("Relative EO change (%):", (eo2-eo1)/abs(eo1)*100)

# Example after reviewing the printed table:
# compare_thresholds(0.25, 0.45)

# -------------------------
# 9. Extra outputs for report writing
# -------------------------
print("\nREPORT SUMMARY")
print(f"Seed={SEED}; train={len(X_train)}; test={len(X_test)}")
print(f"OLS median={median_ols:.6f}; mismatch={mismatch}/{len(y_test)} ({mismatch_pct:.3f}%)")
print(f"Logistic accuracy={accuracy_score(y_test, logit_label):.4f}; ROC-AUC={roc_auc_score(y_test, logit_prob):.4f}")
print(f"High-confidence wrong predictions={wrong_total}/{len(test_results)} ({wrong_pct:.3f}%)")
if len(omit_df) and omit_df["absolute_pct_change"].notna().any():
    top = omit_df.loc[omit_df["absolute_pct_change"].idxmax()]
    print("Largest absolute coefficient percentage shift:", top["predictor"], f'{top["absolute_pct_change"]:.3f}%')
print("\nOutput files:")
for p in sorted(OUT.glob("*")):
    print("-", p.resolve())

print("""
IMPORTANT REPORT NOTES:
1. Part 1 confusion matrix rows are OLS threshold labels and columns are logistic labels,
   not actual outcomes.
2. The subgroup/omitted-variable formula models are inferential models on raw features,
   separate from the preprocessed sklearn predictive models.
3. In this script, the omitted variable is education_num; predictors compared are age,
   hours_per_week, capital_gain, and capital_loss.
4. Fairness ratios are Female/Male. A ratio of 1 is parity; NaN means undefined because
   a denominator was zero or no positive predictions existed.
5. For percentage coefficient change, the script uses (reduced-full)/abs(full)*100.
6. The dataset is historical and does not establish that these features cause income.
""")
