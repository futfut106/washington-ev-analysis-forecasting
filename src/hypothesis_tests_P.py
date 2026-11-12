"""Kiểm định giả thuyết H1, H2, H3, H4a trên dataset [P] (Electric Vehicle Population Data).

Chạy:  python src/hypothesis_tests_P.py
Kết quả in ra màn hình và lưu vào results/P_hypothesis_tests.csv
Yêu cầu: pandas, numpy, scipy
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):  # tránh lỗi mã hóa console trên Windows (bỏ qua trong Jupyter)
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "EV_Population_Washington_Clean__1_.csv"
OUT_PATH = ROOT / "results" / "P_hypothesis_tests.csv"
ALPHA = 0.05

RANGE, YEAR = "Electric Range", "Model Year"
TYPE, CAFV, COUNTY = "Electric Vehicle Type", "Clean Alternative Fuel Vehicle (CAFV) Eligibility", "County"

rows = []  # bảng tổng hợp kết quả


def record(**kw):
    rows.append(kw)


def decision(p):
    return "Bác bỏ H0" if p < ALPHA else "Không bác bỏ H0"


def cramers_v(table):
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = np.asarray(table).sum()
    k = min(np.asarray(table).shape) - 1
    return float(np.sqrt(chi2 / (n * k)))


def load():
    P = pd.read_csv(DATA_PATH)
    for c in [COUNTY, TYPE, CAFV]:
        P[c] = P[c].astype("string").str.strip()
    P["Type"] = np.where(P[TYPE].str.contains("BEV"), "BEV", "PHEV")
    return P


# ---------------------------------------------------------------- H1
def mann_whitney(df, label):
    bev = df.loc[df["Type"] == "BEV", RANGE].to_numpy(float)
    phev = df.loc[df["Type"] == "PHEV", RANGE].to_numpy(float)
    u1, p = stats.mannwhitneyu(bev, phev, alternative="two-sided", method="asymptotic")
    n1, n2 = len(bev), len(phev)
    cles = u1 / (n1 * n2)          # P(BEV > PHEV) + 0.5 P(=)  (xác suất ưu thế)
    r_rb = 2 * cles - 1            # rank-biserial: >0 nghĩa là BEV có range cao hơn
    print(f"[H1 {label}] n_BEV={n1:,} n_PHEV={n2:,} | median BEV={np.median(bev):.0f}, "
          f"PHEV={np.median(phev):.0f} | U={u1:,.0f} p={p:.3g} | rank-biserial={r_rb:.3f} | "
          f"xác suất ưu thế={cles:.3f} -> {decision(p)}")
    record(test=f"H1 Mann-Whitney U ({label})", n=n1 + n2, statistic=u1, p_value=p,
           effect_size=r_rb, effect_name="rank-biserial", extra=f"CLES={cles:.3f}", decision=decision(p))


def h1(P):
    d = P.dropna(subset=[RANGE])
    print(f"\n=== H1: Electric Range theo loại xe | n có Range = {len(d):,} "
          f"(thiếu {P[RANGE].isna().sum():,}) ===")
    mann_whitney(d, "chính, toàn mẫu có Range")
    mann_whitney(d[d[YEAR] <= 2020], "độ nhạy, Model Year<=2020")


# ---------------------------------------------------------------- H2
def spearman(df, label):
    rho, p = stats.spearmanr(df[YEAR], df[RANGE])
    print(f"[H2 {label}] n={len(df):,} rho={rho:.3f} |rho|={abs(rho):.3f} p={p:.3g} -> {decision(p)}")
    record(test=f"H2 Spearman ({label})", n=len(df), statistic=rho, p_value=p,
           effect_size=abs(rho), effect_name="|rho|", extra="", decision=decision(p))


def h2(P):
    d = P.dropna(subset=[RANGE, YEAR])
    print("\n=== H2: Model Year vs Electric Range ===")
    spearman(d, "chính, toàn mẫu")
    for t in ["BEV", "PHEV"]:
        spearman(d[d["Type"] == t], t)


# ---------------------------------------------------------------- H3
def chi_square(table, label, test_name):
    chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
    v = cramers_v(table)
    print(f"[{test_name} {label}] bảng {table.shape[0]}x{table.shape[1]} n={table.to_numpy().sum():,} "
          f"chi2={chi2:,.2f} dof={dof} p={p:.3g} Cramér's V={v:.3f} | min E={exp.min():,.1f} "
          f"-> {decision(p)}")
    record(test=f"{test_name} chi-square ({label})", n=int(table.to_numpy().sum()), statistic=chi2,
           p_value=p, effect_size=v, effect_name="Cramer's V", extra=f"dof={dof}; minE={exp.min():.1f}",
           decision=decision(p))
    return exp


def h3(P):
    print("\n=== H3: Loại xe x CAFV Eligibility ===")
    unk = P[CAFV].str.contains("unknown", case=False, na=False)
    print(f"Nhãn unknown: {unk.sum():,} dòng | trùng với Range thiếu: "
          f"{(unk & P[RANGE].isna()).sum():,} | thuộc BEV: {(unk & (P['Type'] == 'BEV')).sum():,}")
    main = pd.crosstab(P.loc[~unk, "Type"], P.loc[~unk, CAFV])
    print(main)
    chi_square(main, "chính 2x2, bỏ unknown", "H3")
    full = pd.crosstab(P["Type"], P[CAFV])
    print(full)
    chi_square(full, "độ nhạy 2x3, giữ unknown", "H3")


# ---------------------------------------------------------------- H4a
def h4a(P):
    print("\n=== H4a: County x loại xe ===")
    tab = pd.crosstab(P[COUNTY], P["Type"])
    exp = chi_square(tab, f"{tab.shape[0]} county", "H4a")
    pct = (exp < 5).sum() / exp.size
    print(f"Ô có tần số kỳ vọng < 5: {(exp < 5).sum()}/{exp.size} ({pct:.1%})")
    small = tab.sum(axis=1) < 30
    print("County < 30 xe:", tab.index[small].tolist())
    chi_square(tab[~small], f"độ nhạy, bỏ county < 30 xe ({(~small).sum()} county)", "H4a")
    share = (tab["BEV"] / tab.sum(axis=1)).sort_values()
    print("Tỷ lệ BEV thấp nhất:", share.head(3).round(3).to_dict())
    print("Tỷ lệ BEV cao nhất:", share.tail(3).round(3).to_dict())


def main():
    P = load()
    print(f"Dataset [P]: {P.shape[0]:,} dòng | Range thiếu: {P[RANGE].isna().sum():,} | "
          f"Range = 0: {(P[RANGE] == 0).sum():,} | số county: {P[COUNTY].nunique()}")
    print(P["Type"].value_counts().to_dict())
    h1(P)
    h2(P)
    h3(P)
    h4a(P)
    OUT_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu: {OUT_PATH}")


if __name__ == "__main__":
    main()
