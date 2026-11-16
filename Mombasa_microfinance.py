"""
Loan repayment analysis for a (fictional) Mombasa microfinance institution.

Steps:
  1. Generate 150 fictional loan records (reproducible via fixed seed)
  2. Summary statistics (mean, median, std, quartiles)
  3. Outlier detection on loan amount using the IQR method
  4. Correlation between loan amount and days to first repayment
  5. Histogram of loan amounts

Run:  python loan_analysis.py
Outputs: loans.csv, loan_amount_histogram.png
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from scipy import stats

SEED = 42
N = 150
COUNTIES = ["Mombasa", "Kilifi", "Kwale", "Taita-Taveta", "Lamu", "Tana River"]
COUNTY_WEIGHTS = [0.40, 0.20, 0.15, 0.10, 0.08, 0.07]

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- 1. Data
# Most microloans are small; a right-skewed (log-normal) shape is realistic.
amount = rng.lognormal(mean=10.6, sigma=0.55, size=N)  # median ~ KES 40,000
amount = np.clip(amount, 5_000, 400_000)

# Add a few deliberately extreme loans so the outlier check has something to find
amount[[17, 88, 121]] = [310_000, 365_000, 280_000]  # unusually large
amount[[44, 103]] = [2_500, 3_000]  # unusually small
amount = (np.round(amount / 500) * 500).astype(int)  # round to nearest KES 500

# Days to first repayment: ~30 days typical, with a mild upward drift for
# bigger loans plus random noise (so any link is real but weak).
days = 30 + 0.00003 * amount + rng.normal(0, 6, size=N)
days = np.clip(np.round(days), 7, 90).astype(int)

df = pd.DataFrame(
    {
        "loan_id": [f"LN{str(i).zfill(4)}" for i in range(1, N + 1)],
        "amount_kes": amount,
        "days_to_first_repayment": days,
        "county": rng.choice(COUNTIES, size=N, p=COUNTY_WEIGHTS),
    }
)
df.to_csv("loans.csv", index=False)
print(f"Created {len(df)} loan records -> loans.csv\n")

# ------------------------------------------------------ 2. Summary stats
print("=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
cols = ["amount_kes", "days_to_first_repayment"]
summary = pd.DataFrame(
    {
        c: {
            "count": df[c].count(),
            "mean": df[c].mean(),
            "median": df[c].median(),
            "std": df[c].std(),
            "min": df[c].min(),
            "Q1 (25%)": df[c].quantile(0.25),
            "Q3 (75%)": df[c].quantile(0.75),
            "max": df[c].max(),
        }
        for c in cols
    }
)
print(summary.round(2).to_string())
print()

# ----------------------------------------------------- 3. IQR outliers
q1 = df["amount_kes"].quantile(0.25)
q3 = df["amount_kes"].quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

high = df[df["amount_kes"] > upper_fence]
low = df[df["amount_kes"] < lower_fence]

print("=" * 60)
print("OUTLIERS (IQR METHOD) - loan amount")
print("=" * 60)
print(f"Q1 = {q1:,.0f}   Q3 = {q3:,.0f}   IQR = {iqr:,.0f}")
print(f"Lower fence = {lower_fence:,.0f}   Upper fence = {upper_fence:,.0f}\n")
print(f"Unusually LARGE loans ({len(high)}):")
for _, r in high.iterrows():
    print(f"  {r.loan_id}  KES {r.amount_kes:>9,}  ({r.county})")
print(f"Unusually SMALL loans ({len(low)}):")
if len(low):
    for _, r in low.iterrows():
        print(f"  {r.loan_id}  KES {r.amount_kes:>9,}  ({r.county})")
else:
    print("  none (lower fence is below zero, so no loan can fall under it)"
          if lower_fence < 0 else "  none")
print(f"\nOutlier loan IDs: {', '.join(pd.concat([low, high]).loan_id)}\n")

# ---------------------------------------------------- 4. Correlation
r_p, p_p = stats.pearsonr(df["amount_kes"], df["days_to_first_repayment"])
r_s, p_s = stats.spearmanr(df["amount_kes"], df["days_to_first_repayment"])

print("=" * 60)
print("CORRELATION: loan amount vs days to first repayment")
print("=" * 60)
print(f"Pearson  r = {r_p:+.3f}  (p = {p_p:.4f})")
print(f"Spearman r = {r_s:+.3f}  (p = {p_s:.4f})  <- robust to outliers")


def describe(r: float, p: float) -> str:
    strength = (
        "no meaningful" if abs(r) < 0.1
        else "a weak" if abs(r) < 0.3
        else "a moderate" if abs(r) < 0.5
        else "a strong"
    )
    direction = "positive" if r > 0 else "negative"
    sig = "statistically significant" if p < 0.05 else "not statistically significant"
    return f"{strength} {direction} link (r = {r:+.2f}, {sig})"


print(f"\nIn plain terms: the data show {describe(r_p, p_p)}.\n")

# ------------------------------------------------------- 5. Histogram
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.hist(df["amount_kes"], bins=20, color="#2a6f97", edgecolor="white")
ax.axvline(df["amount_kes"].mean(), color="#d1495b", linestyle="--", linewidth=1.8,
           label=f"Mean: KES {df['amount_kes'].mean():,.0f}")
ax.axvline(df["amount_kes"].median(), color="#2e933c", linestyle="-", linewidth=1.8,
           label=f"Median: KES {df['amount_kes'].median():,.0f}")
ax.axvline(upper_fence, color="#6c757d", linestyle=":", linewidth=1.8,
           label=f"Outlier threshold: KES {upper_fence:,.0f}")
ax.set_title("Distribution of Loan Amounts (150 fictional loans)", fontsize=14, pad=12)
ax.set_xlabel("Loan amount (KES)")
ax.set_ylabel("Number of loans")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("loan_amount_histogram.png", dpi=150)
print("Saved histogram -> loan_amount_histogram.png")
