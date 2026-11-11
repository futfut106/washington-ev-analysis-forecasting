"""Tạo thống kê mô tả cho mục 2.2.1 từ CSV Population đã làm sạch."""

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(population):
    columns = [
        "Electric Vehicle Type", "Make", "Model", "Model Year",
        "Electric Range", "Base MSRP",
    ]
    missing_columns = sorted(set(columns) - set(population.columns))
    if missing_columns:
        raise ValueError(f"Thiếu cột: {', '.join(missing_columns)}")
    if population.empty:
        raise ValueError("Population không có bản ghi.")

    total = len(population)
    summary = {"total": total, "attributes": {}, "vehicle_types": {}}
    for name in columns:
        values = population[name]
        missing = int(values.isna().sum())
        summary["attributes"][name] = {
            "distinct": int(values.nunique()),
            "valid": total - missing,
            "missing": missing,
            "missing_pct": 100 * missing / total,
        }

    for name, group in population.groupby("Electric Vehicle Type", dropna=False):
        missing_range = int(group["Electric Range"].isna().sum())
        summary["vehicle_types"][str(name)] = {
            "count": len(group),
            "share_pct": 100 * len(group) / total,
            "missing_range": missing_range,
            "missing_range_pct": 100 * missing_range / len(group),
        }

    summary["distinct_make_model_pairs"] = len(
        population[["Make", "Model"]].dropna().drop_duplicates()
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("population", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.population, encoding="utf-8-sig")
    print(json.dumps(summarize(data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
