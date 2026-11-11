"""Chương 3, mục 3.3: thiết kế tập dữ liệu, KHÔNG huấn luyện mô hình.

Cách chạy (Python 3 + pandas + numpy):
python ch3_3_3.py --population "EV_Population_Washington_Clean (1).csv" \
    --history "EV_Population_History_Washington_Clean.csv" --output du_lieu_3_3

Mặc định tìm hai CSV trong thư mục hiện tại hoặc thư mục chứa script.
Tập kiểm tra chỉ được dùng làm đích đánh giá. Với dự báo 12 tháng từ một
gốc dự báo cố định (11/2022), các lag sau gốc đó phải lấy từ dự báo trước,
không lấy từ giá trị thực của tập kiểm tra.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATE = "Date"
COUNTY = "County"
USE = "Vehicle Primary Use"
BEV = "Battery Electric Vehicles (BEVs)"
PHEV = "Plug-In Hybrid Electric Vehicles (PHEVs)"
EV = "Electric Vehicle (EV) Total"
NON_EV = "Non-Electric Vehicle Total"
TOTAL = "Total Vehicles"
PCT = "Percent Electric Vehicles"
TYPE = "Electric Vehicle Type"
RANGE = "Electric Range"
CAFV = "Clean Alternative Fuel Vehicle (CAFV) Eligibility"
UNKNOWN = "Eligibility unknown as battery range has not been researched"
COUNT_COLUMNS = [BEV, PHEV, EV, NON_EV, TOTAL]
LAGS = (1, 2, 3, 6, 12)
TEST_START = pd.Period("2022-12", freq="M")


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def default_input(pattern: str) -> Path | None:
    candidates: list[Path] = []
    for directory in (Path.cwd(), Path(__file__).resolve().parent):
        candidates.extend(directory.glob(pattern))
    candidates = list(dict.fromkeys(p.resolve() for p in candidates))
    return candidates[0] if len(candidates) == 1 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(frame: pd.DataFrame) -> list[dict]:
    """JSON records with dates serialized and missing numbers as null."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")


def prepare(population_path: Path, history_path: Path, output_dir: Path) -> dict:
    p = pd.read_csv(population_path)
    h = pd.read_csv(history_path)
    p_columns = [
        COUNTY, "City", "State", "Postal Code", "Model Year", "Make", "Model",
        TYPE, CAFV, RANGE, "Base MSRP",
    ]
    h_columns = [DATE, COUNTY, "State", USE, *COUNT_COLUMNS, PCT]
    require(list(p.columns) == p_columns, "Population phải có đúng 11 cột theo phiên bản của nhóm.")
    require(list(h.columns) == h_columns, "History phải có đúng 10 cột theo phiên bản của nhóm.")
    require(p.shape == (134726, 11), "Quy mô Population khác phiên bản đã phân công (134.726 × 11).")
    require(h.shape == (6474, 10), "Quy mô History khác phiên bản đã phân công (6.474 × 10).")
    require(p["State"].eq("WA").all() and h["State"].eq("WA").all(), "Có bản ghi ngoài WA.")
    require(h.notna().all().all(), "History có giá trị thiếu: cần kiểm tra trước tổng hợp.")
    h[DATE] = pd.to_datetime(h[DATE], format="%Y-%m-%d", errors="raise")
    require(h[DATE].dt.is_month_end.all(), "Date phải là ngày cuối tháng.")
    require(not h.duplicated([DATE, COUNTY, USE]).any(), "History trùng khóa tháng × county × nhóm sử dụng.")
    require(h[COUNTY].nunique() == 39 and h[USE].nunique() == 2, "Cấu trúc county/nhóm sử dụng khác dự kiến.")
    require(set(h[USE]) == {"Passenger", "Truck"}, "Nhãn Vehicle Primary Use khác dự kiến.")
    require((h[COUNT_COLUMNS] >= 0).all().all(), "Số lượng xe không được âm.")
    require(np.equal(h[COUNT_COLUMNS], np.floor(h[COUNT_COLUMNS])).all().all(), "Có số lượng xe không nguyên.")
    require(h[BEV].add(h[PHEV]).eq(h[EV]).all(), "BEV + PHEV khác EV Total.")
    require(h[EV].add(h[NON_EV]).eq(h[TOTAL]).all(), "EV + Non-Electric khác Total Vehicles.")
    require(h[TOTAL].gt(0).all(), "Mẫu số Total Vehicles phải dương.")
    expected_pct = (100 * h[EV] / h[TOTAL]).round(2)
    require(np.allclose(h[PCT], expected_pct, atol=1e-9, rtol=0), "Percent Electric Vehicles không khớp tỷ lệ làm tròn.")
    dates = pd.date_range("2017-01-31", "2023-11-30", freq="ME")
    require(set(h[DATE]) == set(dates), "History phải có 83 tháng liên tục từ 01/2017 đến 11/2023.")
    require(h.groupby(DATE).size().eq(78).all(), "Một tháng không đủ 39 × 2 dòng.")
    require(h.groupby([DATE, COUNTY])[USE].nunique().eq(2).all(), "Một county-tháng thiếu Passenger hoặc Truck.")
    h[COUNT_COLUMNS] = h[COUNT_COLUMNS].astype("int64")

    # Tập 1 [P]: giữ nguyên bản ghi; không drop_duplicates trên 11 thuộc tính.
    valid_range = p[RANGE].notna()
    p_known_cafv = ~p[CAFV].eq(UNKNOWN)

    # Tập 2: một county × một tháng, đã cộng Passenger và Truck.
    county = h.groupby([DATE, COUNTY], as_index=False)[COUNT_COLUMNS].sum()
    county[PCT] = 100 * county[EV] / county[TOTAL]
    county_cuts = county[county[DATE].isin([dates[0], dates[-1]])].sort_values([DATE, COUNTY]).reset_index(drop=True)
    require(len(county_cuts) == 78, "Hai lát cắt county phải có 39 + 39 = 78 dòng.")
    require(county_cuts.groupby(DATE).size().eq(39).all(), "Một lát cắt thiếu county.")

    # Tập 3: chuỗi toàn bang. Chỉ cộng các cột số lượng cùng một tháng.
    monthly = h.groupby(DATE, as_index=False)[COUNT_COLUMNS].sum().sort_values(DATE).reset_index(drop=True)
    monthly[PCT] = 100 * monthly[EV] / monthly[TOTAL]
    require(len(monthly) == 83, "Chuỗi toàn bang phải có 83 điểm.")
    require(monthly[DATE].tolist() == dates.tolist(), "Chỉ số tháng bị đứt hoặc sai thứ tự.")
    train = monthly[monthly[DATE].dt.to_period("M") < TEST_START].copy()
    test = monthly[monthly[DATE].dt.to_period("M") >= TEST_START].copy()
    require(len(train) == 71 and len(test) == 12, "Thiết kế phải là 71 tháng train và 12 tháng test.")
    require(train[DATE].max() < test[DATE].min(), "Train và test không được chồng thời gian.")
    require(train[DATE].max() == pd.Timestamp("2022-11-30"), "Mốc kết thúc train không đúng.")
    require(test[DATE].min() == pd.Timestamp("2022-12-31"), "Mốc bắt đầu test không đúng.")

    # Chỉ tạo đặc trưng huấn luyện từ chuỗi train: y_t là nhãn, lag là quá khứ.
    rf_train = train[[DATE, EV]].rename(columns={EV: "y"}).copy()
    for lag in LAGS:
        rf_train[f"lag_{lag}"] = rf_train["y"].shift(lag)
    rf_train["month"] = rf_train[DATE].dt.month
    rf_train = rf_train.dropna(subset=[f"lag_{lag}" for lag in LAGS]).reset_index(drop=True)
    require(len(rf_train) == 59, "Lag 12 phải để lại 71 - 12 = 59 dòng huấn luyện.")
    require(rf_train[DATE].min() == pd.Timestamp("2018-01-31"), "Dòng RF đầu tiên phải là 01/2018.")
    rf_train[[f"lag_{lag}" for lag in LAGS]] = rf_train[[f"lag_{lag}" for lag in LAGS]].astype("int64")
    history_by_month = train.assign(period=train[DATE].dt.to_period("M")).set_index("period")[EV]

    # Template cho dự báo nhiều bước từ một gốc cố định. Không đọc y_test để điền lag.
    future_rows = []
    for step, date in enumerate(test[DATE], start=1):
        period = date.to_period("M")
        row = {DATE: date, "forecast_step": step, "month": date.month}
        for lag in LAGS:
            source = period - lag
            known = source in history_by_month.index
            row[f"lag_{lag}"] = int(history_by_month.loc[source]) if known else None
            row[f"lag_{lag}_source"] = ("observed_train:" if known else "prediction_required:") + str(source)
            require(source < period, "Đặc trưng lag không được trỏ đến hiện tại/tương lai.")
            if known:
                require(source < TEST_START, "Giá trị lag ban đầu không được lấy từ test.")
        future_rows.append(row)
    rf_future = pd.DataFrame(future_rows)
    test_targets = test[[DATE, EV]].rename(columns={EV: "y_actual"}).copy()
    split = monthly[[DATE]].copy()
    split["split"] = np.where(monthly[DATE].dt.to_period("M") < TEST_START, "train", "test")
    split["rf_training_row"] = split[DATE].isin(rf_train[DATE])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(monthly, output_dir / "statewide_monthly_83.csv")
    write_csv(county_cuts, output_dir / "county_H4b_two_snapshots_78.csv")
    write_csv(train[[DATE, EV]].rename(columns={EV: "y"}), output_dir / "series_train_71.csv")
    write_csv(test_targets, output_dir / "test_targets_12.csv")
    write_csv(rf_train, output_dir / "rf_train_features_59.csv")
    write_csv(rf_future, output_dir / "rf_forecast_design_12.csv")
    write_csv(split, output_dir / "temporal_split_83.csv")
    evidence = {
        "scope": "Data design only; no forecasting or hypothesis models fitted.",
        "sources": {
            "population": {"filename": population_path.name, "sha256": sha256(population_path)},
            "history": {"filename": history_path.name, "sha256": sha256(history_path)},
        },
        "population": {
            "rows": len(p), "columns": len(p.columns), "county_count": p[COUNTY].nunique(),
            "city_name_count_before_normalization": p["City"].nunique(),
            "exact_duplicate_attribute_rows_kept": int(p.duplicated().sum()),
            "valid_range_rows_H1_H2": int(valid_range.sum()),
            "valid_range_by_type": p.loc[valid_range, TYPE].value_counts().to_dict(),
            "known_CAFV_rows_H3_main": int(p_known_cafv.sum()),
            "full_CAFV_rows_H3_sensitivity": len(p),
            "columns_exact": list(p.columns),
        },
        "history": {"rows": len(h), "columns": len(h.columns), "county_count": 39,
                    "primary_use_count": 2, "month_count": 83, "county_snapshot_rows": 78,
                    "county_rows_per_snapshot": 39, "columns_exact": list(h.columns)},
        "split": {
            "train_rows": len(train), "test_rows": len(test),
            "train_start": str(train[DATE].min().date()), "train_end": str(train[DATE].max().date()),
            "test_start": str(test[DATE].min().date()), "test_end": str(test[DATE].max().date()),
            "seasonal_cycles_train": len(train) / 12,
            "RF_train_effective_rows": len(rf_train),
            "RF_train_start": str(rf_train[DATE].min().date()),
            "RF_train_end": str(rf_train[DATE].max().date()),
            "RF_predictors": [f"lag_{lag}" for lag in LAGS] + ["month"],
            "RF_unknown_test_lag_cells": int(rf_future[[f"lag_{lag}" for lag in LAGS]].isna().sum().sum()),
            "forecast_protocol": "Fixed origin at 2022-11-30, recursive 12-step forecasts. For a lag whose source month is in the test period, substitute an earlier forecast, never the actual test target.",
        },
        "sample_months": records(monthly[monthly[DATE].isin([dates[0], train[DATE].max(), test[DATE].min(), dates[-1]])]),
        "rf_training_examples": records(rf_train.iloc[[0, -1]]),
        "rf_forecast_examples": records(rf_future.iloc[[0, 1, -1]]),
        "county_snapshot_state_totals": records(county_cuts.groupby(DATE, as_index=False)[COUNT_COLUMNS].sum()),
        "checks": {
            "unique_history_key": True, "complete_month_county_use_grid": True,
            "no_missing_history_values": True, "count_reconciliations_exact": True,
            "source_percent_matches_rounded_ratio": True,
            "train_test_no_overlap": True, "training_lags_strictly_past": True,
            "test_lags_never_initialized_from_actual_test_targets": True,
        },
    }
    evidence = json.loads(json.dumps(evidence, ensure_ascii=False, default=lambda obj: obj.item() if hasattr(obj, "item") else str(obj)))
    (output_dir / "ch3_3_3_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population", type=Path, default=default_input("EV_Population_Washington_Clean*.csv"))
    parser.add_argument("--history", type=Path, default=default_input("EV_Population_History_Washington_Clean.csv"))
    parser.add_argument("--output", type=Path, default=Path("du_lieu_3_3"))
    args = parser.parse_args()
    if args.population is None or args.history is None:
        parser.error("Không tìm thấy duy nhất hai CSV. Hãy truyền --population và --history.")
    result = prepare(args.population, args.history, args.output)
    print(json.dumps({"population_rows": result["population"]["rows"], "history_rows": result["history"]["rows"],
                      "monthly_rows": result["history"]["month_count"], "split": result["split"],
                      "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
