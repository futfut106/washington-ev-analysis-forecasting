"""Kiểm định giả thuyết H4b trên dataset [H] (số xe điện / tổng số xe theo county, theo tháng).

Chạy:  python src/hypothesis_tests_H.py
Kết quả in ra màn hình và lưu vào results/H_hypothesis_tests.csv
Yêu cầu: pandas, numpy, scipy

H0: tỷ lệ EV = EV Total / Total Vehicles như nhau ở mọi county.
H1: có ít nhất một county có tỷ lệ khác.
Phép kiểm định: chi-square đồng nhất tỷ lệ trên bảng county x (EV, non-EV), kèm Cramér's V.
Mỗi lần chạy dùng một tháng cắt ngang (tháng cuối và tháng đầu) để tránh đếm lặp cùng một xe.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):  # tránh lỗi mã hóa console trên Windows (bỏ qua trong Jupyter)
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
# File [H] hiện đang tên Electric_Vehicle_Population_Data.csv (đổi tên thành [H]_raw.csv thì sửa ở đây)
DATA_PATH = ROOT / "data" / "Electric_Vehicle_Population_Data.csv"
OUT_PATH = ROOT / "results" / "H_hypothesis_tests.csv"
ALPHA = 0.05


def decision(p):
    return "Bác bỏ H0" if p < ALPHA else "Không bác bỏ H0"


def run_month(H, month, rows):
    d = H[H["Date"] == month]
    # cộng Passenger và Truck theo county
    g = d.groupby("County")[["Electric Vehicle (EV) Total", "Non-Electric Vehicle Total"]].sum()
    g = g[g.sum(axis=1) > 0]
    table = g.round().astype("int64")
    chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
    n = int(table.to_numpy().sum())
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    share = (table.iloc[:, 0] / table.sum(axis=1)).sort_values()
    label = f"{month:%m/%Y}"
    print(f"[H4b {label}] {table.shape[0]} county x 2 | n={n:,} | chi2={chi2:,.1f} dof={dof} "
          f"p={p:.3g} | Cramér's V={v:.3f} | min E={exp.min():,.1f} | tỷ lệ EV chung={table.iloc[:,0].sum()/n:.3%} "
          f"-> {decision(p)}")
    print("  Tỷ lệ EV thấp nhất:", {k: f"{x:.2%}" for k, x in share.head(3).items()})
    print("  Tỷ lệ EV cao nhất:", {k: f"{x:.2%}" for k, x in share.tail(3).items()})
    rows.append(dict(test=f"H4b chi-square ({label})", n=n, statistic=chi2, p_value=p,
                     effect_size=v, effect_name="Cramer's V", extra=f"dof={dof}; minE={exp.min():.1f}",
                     decision=decision(p)))


def main():
    H = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    H["County"] = H["County"].str.strip()
    print(f"Dataset [H]: {H.shape[0]:,} dòng | {H['Date'].nunique()} tháng "
          f"({H['Date'].min():%m/%Y} - {H['Date'].max():%m/%Y}) | {H['County'].nunique()} county | "
          f"Vehicle Primary Use: {sorted(H['Vehicle Primary Use'].dropna().unique())}")
    print("Cột thiếu:", H.isna().sum()[H.isna().sum() > 0].to_dict())
    # kiểm tra tính nhất quán số liệu
    ev_ok = (H["Battery Electric Vehicles (BEVs)"] + H["Plug-In Hybrid Electric Vehicles (PHEVs)"]
             == H["Electric Vehicle (EV) Total"]).mean()
    tot_ok = (H["Electric Vehicle (EV) Total"] + H["Non-Electric Vehicle Total"]
              == H["Total Vehicles"]).mean()
    print(f"BEV+PHEV=EV Total: {ev_ok:.2%} | EV+NonEV=Total: {tot_ok:.2%}\n")

    rows = []
    print("=== H4b: tỷ lệ EV giữa các county ===")
    run_month(H, H["Date"].max(), rows)   # tháng cuối (11/2023)
    run_month(H, H["Date"].min(), rows)   # tháng đầu (01/2017)
    OUT_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu: {OUT_PATH}")


if __name__ == "__main__":
    main()
