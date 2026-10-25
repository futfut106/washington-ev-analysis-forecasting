"""Preprocessing pipeline for dataset [P] - Electric Vehicle Population Data (Washington State).

Run:   python src/preprocess_P.py
       python src/preprocess_P.py --input data/EV_Population_Washington_Clean__1_.csv \
                                  --output data/processed/EV_Population_WA_Processed.csv

Design principles (taken from report Group 8, Chapter 3 "Data and preprocessing", section 3.2):
  * Population [P] is a CROSS-SECTIONAL snapshot of registered vehicles (no date column).
  * Missing Electric Range / Base MSRP is NEVER replaced by 0 or imputed; it is kept as NA.
  * Rows are NEVER dropped: the clean file has no DOL Vehicle ID, so identical rows cannot be
    proven to be duplicates. Duplicates are only reported.
  * Original column names are preserved so src/hypothesis_tests_P.py keeps working; derived
    columns are only ADDED.

Every step documents WHAT it does, HOW it is done and WHY it is needed.
Requires: pandas (>=2.0), numpy
Outputs : data/processed/EV_Population_WA_Processed.csv  (cleaned data)
          results/P_preprocessing_audit.csv              (audit trail of every step)
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):  # avoid console encoding errors on Windows
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "EV_Population_Washington_Clean__1_.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "EV_Population_WA_Processed.csv"
AUDIT_OUT = ROOT / "results" / "P_preprocessing_audit.csv"

# ----------------------------------------------------------------------------- column names
COUNTY, CITY, STATE, ZIP = "County", "City", "State", "Postal Code"
YEAR, MAKE, MODEL = "Model Year", "Make", "Model"
TYPE_RAW = "Electric Vehicle Type"
CAFV = "Clean Alternative Fuel Vehicle (CAFV) Eligibility"
RANGE, MSRP = "Electric Range", "Base MSRP"

EXPECTED_COLUMNS = [COUNTY, CITY, STATE, ZIP, YEAR, MAKE, MODEL, TYPE_RAW, CAFV, RANGE, MSRP]
TEXT_COLUMNS = [COUNTY, CITY, STATE, MAKE, MODEL, TYPE_RAW, CAFV]

# ----------------------------------------------------------------------------- business rules
STUDY_STATE = "WA"
YEAR_MIN, YEAR_MAX = 1980, date.today().year + 1   # plausible Model Year window
RANGE_MAX_MILES = 500                               # no production EV/PHEV exceeds this
WA_ZIP_MIN, WA_ZIP_MAX = 98001, 99403               # Washington ZIP code span
CITY_OVERRIDES = {"MCCLEARY": "McCleary"}           # known mixed-case variants (Mccleary / McCleary)
CAFV_SHORT = {
    "Clean Alternative Fuel Vehicle Eligible": "Eligible",
    "Not eligible due to low battery range": "Not eligible",
    "Eligibility unknown as battery range has not been researched": "Unknown",
}

# Figures stated in the report - used as a regression check at the end of the pipeline.
REPORT_FIGURES = {
    "rows (N)": 134_726, "BEV rows": 103_674, "PHEV rows": 31_052, "distinct County": 39,
    "distinct Make": 36, "distinct Model (non-null)": 125, "distinct Model Year": 22,
    "Electric Range missing": 56_869, "Base MSRP valid": 3_419, "Model missing": 249,
    "CAFV 'Unknown' rows": 56_868,
}

AUDIT: list[dict] = []   # filled by log()


def log(step: str, rows_in: int, rows_out: int, detail: str) -> None:
    """Record one pipeline action in the audit trail and echo it to the console."""
    AUDIT.append({"step": step, "rows_in": rows_in, "rows_out": rows_out, "detail": detail})
    print(f"[{step}] rows {rows_in:,} -> {rows_out:,} | {detail}")


# =============================================================================================
# STEP 1 - LOAD
# =============================================================================================
def load_raw(path: Path) -> pd.DataFrame:
    """
    WHAT: Read the CSV into a DataFrame without letting pandas guess the type of 'Postal Code'.
    HOW : pd.read_csv(..., dtype={'Postal Code': 'string'}); every other column uses defaults.
    WHY : The CSV stores ZIP codes as '99344.0' (float artefact). If pandas parsed them as float
          the value could lose its leading zeros / become a number; reading as text preserves
          the raw value so that step 5 can repair it deterministically.
    """
    df = pd.read_csv(path, dtype={ZIP: "string"})
    log("1 load", len(df), len(df), f"{df.shape[1]} columns from {path.name}")
    return df


# =============================================================================================
# STEP 2 - SCHEMA VALIDATION
# =============================================================================================
def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Verify that exactly the 11 columns described in the report are present.
    HOW : Compare the column set with EXPECTED_COLUMNS; raise ValueError on missing columns and
          drop unexpected extras (e.g. columns re-introduced from the 17-column raw file).
    WHY : Every later step addresses columns by name. Failing fast on a wrong file is cheaper
          than silently producing a wrong analysis; keeping only the 11 agreed columns keeps the
          output identical to the dataset documented in Table 2.1 / 3.1.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    df = df[EXPECTED_COLUMNS].copy()
    log("2 schema", len(df), len(df), f"11 columns OK; dropped extras: {extra or 'none'}")
    return df


# =============================================================================================
# STEP 3 - SCOPE FILTER
# =============================================================================================
def filter_scope(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Keep only vehicles registered with State == 'WA'.
    HOW : Boolean mask on the stripped/upper-cased State column.
    WHY : Vehicles registered in Washington can be physically located in other states, which are
          outside the research scope (report 1.4 / 3.2). In the clean file this removes 0 rows,
          but the guard keeps the pipeline safe if it is ever run on the 17-column raw file.
    """
    n0 = len(df)
    mask = df[STATE].astype("string").str.strip().str.upper() == STUDY_STATE
    df = df[mask].copy()
    log("3 scope", n0, len(df), f"removed {n0 - len(df):,} rows with State != {STUDY_STATE}")
    return df


# =============================================================================================
# STEP 4 - TEXT STANDARDISATION
# =============================================================================================
def standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Normalise whitespace and letter-case in all categorical text columns and unify
          spelling variants of the same City.
    HOW : (a) strip + collapse repeated spaces in every text column;
          (b) Make and Model -> UPPER CASE;
          (c) City: group names by their case-insensitive key and replace every variant by one
              canonical spelling (CITY_OVERRIDES first, otherwise the most frequent variant);
          (d) validate Electric Vehicle Type / CAFV against the allowed label sets.
    WHY : Without this, 'Tesla' vs 'TESLA' or 'Mccleary' vs 'McCleary' would be counted as
          different groups, inflating category counts and distorting Top-N / chi-square tables
          (H3, H4a). Label validation guarantees that a new, unexpected category cannot slip
          into the hypothesis tests unnoticed.
    NOTE: County names are NOT upper-cased: they are already consistent (39 values) and
          the report quotes them in title case.
    """
    out = df.copy()
    changed = {}
    for c in TEXT_COLUMNS:
        before = out[c].astype("string")
        after = before.str.strip().str.replace(r"\s+", " ", regex=True)
        if c in (MAKE, MODEL):
            after = after.str.upper()
        changed[c] = int((before.fillna("") != after.fillna("")).sum())
        out[c] = after

    # City: unify case variants
    key = out[CITY].str.upper()
    counts = (pd.DataFrame({"key": key, "variant": out[CITY]})
              .value_counts().rename("n").reset_index())
    canon = counts.sort_values(["key", "n"], ascending=[True, False]).drop_duplicates("key")
    canon_map = dict(zip(canon["key"], canon["variant"])) | CITY_OVERRIDES
    new_city = key.map(canon_map)
    changed[CITY] += int((out[CITY] != new_city).sum())
    out[CITY] = new_city

    # label validation
    bad_type = ~out[TYPE_RAW].str.extract(r"\((BEV|PHEV)\)")[0].notna()
    bad_cafv = ~out[CAFV].isin(CAFV_SHORT)
    if bad_type.any() or bad_cafv.any():
        raise ValueError(f"Unexpected labels: {bad_type.sum()} vehicle types, {bad_cafv.sum()} CAFV labels")

    log("4 text", len(df), len(out),
        "cells modified per column: "
        + (", ".join(f"{k}={v}" for k, v in changed.items() if v) or "none"))
    log("4 text", len(out), len(out),
        f"distinct City {df[CITY].nunique()} -> {out[CITY].nunique()}; "
        f"Make {out[MAKE].nunique()}; Model {out[MODEL].nunique()}")
    return out


# =============================================================================================
# STEP 5 - TYPE CASTING
# =============================================================================================
def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Give each column its proper data type.
            Postal Code -> 5-character string ('99344.0' -> '99344')
            Model Year, Electric Range, Base MSRP -> nullable integer (Int64)
            Make, Model, County, City, ... -> 'category'-ready text (kept as string)
    HOW : to_numeric(errors='coerce') -> astype('Int64'); ZIP is converted via Int64 then
          zero-filled to width 5.
    WHY : The CSV serialises integers as floats (2020.0, 266.0). A float Model Year would show
          up as '2020.0' in tables and breaks joins/groupings; a float ZIP is a nominal label,
          not a quantity. Nullable Int64 (not int64) is used because Electric Range and Base
          MSRP contain real missing values that must stay NA instead of becoming 0 or NaN-floats.
    """
    out = df.copy()
    zip_num = pd.to_numeric(out[ZIP], errors="coerce").astype("Int64")
    out[ZIP] = zip_num.astype("string").str.zfill(5)
    for c in (YEAR, RANGE, MSRP):
        num = pd.to_numeric(out[c], errors="coerce")
        non_int = int(((num % 1 != 0) & num.notna()).sum())   # fractional values would be truncated
        if non_int:
            raise ValueError(f"{c}: {non_int} non-integer values; refusing to truncate")
        out[c] = num.astype("Int64")
    log("5 types", len(df), len(out), "Postal Code->str(5); Model Year/Electric Range/Base MSRP->Int64")
    return out


# =============================================================================================
# STEP 6 - INVALID VALUE HANDLING
# =============================================================================================
def handle_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Convert impossible / placeholder values to NA and flag suspicious ones.
            * Electric Range == 0 or > 500 mi  -> NA   (DOL uses 0 for 'not researched')
            * Base MSRP <= 0                   -> NA   (0 is a placeholder, not a price)
            * Model Year outside 1980..current+1 -> NA
            * Postal Code not in the Washington ZIP span -> only REPORTED, not changed
    HOW : Boolean masks + .mask(); counts are written to the audit trail.
    WHY : A zero range / price would drag means and rank tests (H1, H2) toward 0 and make 'unknown'
          look like 'zero'. Converting to NA keeps statistics valid-case only, as stated in report
          2.2.1 / 3.2. In the current file all four counts are 0 - the step is a safeguard and
          documents that the check was made. Old models (1997-2008) are NOT removed: they are
          real vehicles (e.g. Tesla Roadster, Ford Ranger EV) and only affect the lower tail.
    """
    out = df.copy()
    r_bad = (out[RANGE] <= 0) | (out[RANGE] > RANGE_MAX_MILES)
    m_bad = out[MSRP] <= 0
    y_bad = (out[YEAR] < YEAR_MIN) | (out[YEAR] > YEAR_MAX)
    out[RANGE] = out[RANGE].mask(r_bad.fillna(False))
    out[MSRP] = out[MSRP].mask(m_bad.fillna(False))
    out[YEAR] = out[YEAR].mask(y_bad.fillna(False))
    zip_int = pd.to_numeric(out[ZIP], errors="coerce")
    zip_out = int((~zip_int.between(WA_ZIP_MIN, WA_ZIP_MAX) | out[ZIP].str.len().ne(5)).sum())
    old = int((out[YEAR] < 2010).sum())
    log("6 invalid", len(df), len(out),
        f"Range->NA {int(r_bad.sum())}; MSRP->NA {int(m_bad.sum())}; Year->NA {int(y_bad.sum())}; "
        f"ZIP outside WA span (reported only) {zip_out}; Model Year<2010 kept {old}")
    return out


# =============================================================================================
# STEP 7 - FEATURE ENGINEERING
# =============================================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Add four derived columns (original columns are untouched):
            Type          'BEV' / 'PHEV'  - short label of Electric Vehicle Type
            CAFV Status   'Eligible' / 'Not eligible' / 'Unknown' - short CAFV label
            Range Missing True when Electric Range is NA
            County City   '<County> | <City>' composite key
    HOW : regex extract '(BEV|PHEV)'; dictionary mapping; isna(); string concatenation.
    WHY : * Type is the grouping variable of H1, H2, H3, H4a - deriving it once avoids every
            script re-implementing (and possibly diverging on) the same string test.
          * CAFV Status makes cross-tabs readable (H3).
          * Range Missing makes the structural missingness (BEV, Model Year >= 2021) explicit and
            lets analysts filter valid cases consistently.
          * County City: 38 city names occur in more than one county (493 pairs for 454 names),
            so statistics by City must use (County, City) to avoid merging different places
            (report 2.2.2).
    """
    out = df.copy()
    out["Type"] = out[TYPE_RAW].str.extract(r"\((BEV|PHEV)\)")[0]
    out["CAFV Status"] = out[CAFV].map(CAFV_SHORT)
    out["Range Missing"] = out[RANGE].isna()
    out["County City"] = out[COUNTY] + " | " + out[CITY]
    log("7 features", len(df), len(out), "added Type, CAFV Status, Range Missing, County City")
    return out


# =============================================================================================
# STEP 8 - MISSING VALUE AUDIT (report only)
# =============================================================================================
def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Quantify missing values per column and per vehicle type, and cross-check them against
          the CAFV 'Unknown' label. Nothing is imputed or dropped.
    HOW : isna().sum(); groupby('Type'); crosstab of Range Missing x CAFV Status.
    WHY : Missingness is NOT random: 54.85% of BEV have no Electric Range versus 1 PHEV, concentrated
          in Model Year >= 2021, and it coincides with the CAFV label 'Unknown' (the DOL has not
          researched the battery range). Imputing (mean/median/0) would invent ranges for newer
          BEVs and bias H1/H2. The policy is therefore: keep NA, analyse valid cases, report the
          valid n for each test, and exclude Base MSRP (97.46% missing) from tests and models.
    """
    n = len(df)
    miss = df[EXPECTED_COLUMNS].isna().sum()
    miss = miss[miss > 0]
    by_type = df.groupby("Type")["Range Missing"].agg(["sum", "size"])
    unk = df["CAFV Status"].eq("Unknown")
    both = int((unk & df["Range Missing"]).sum())
    log("8 missing", n, n,
        "; ".join(f"{c}={v:,} ({v / n:.2%})" for c, v in miss.items()))
    log("8 missing", n, n,
        "Range missing by type: " + ", ".join(f"{t}={int(r['sum']):,}/{int(r['size']):,}"
                                                for t, r in by_type.iterrows())
        + f"; CAFV Unknown & Range missing overlap={both:,} of Unknown={int(unk.sum()):,}")
    return df


# =============================================================================================
# STEP 9 - DUPLICATE AUDIT (report only)
# =============================================================================================
def audit_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Count rows that are identical on all 11 original columns - but DO NOT remove them.
    HOW : DataFrame.duplicated(subset=EXPECTED_COLUMNS) and a group-size summary.
    WHY : The source dataset has one row per registered vehicle, and the clean file has no
          DOL Vehicle ID. Fleet-style models (e.g. 2022 Tesla Model Y in the same city and ZIP)
          legitimately share every attribute. Dropping them would delete real vehicles (about
          70% of the file) and destroy BEV/PHEV and County shares used in RQ1, RQ2, H4a.
          So the decision is: keep every row, document the count.
    """
    n = len(df)
    dup = int(df.duplicated(subset=EXPECTED_COLUMNS).sum())
    biggest = int(df.groupby(EXPECTED_COLUMNS, dropna=False).size().max())
    log("9 duplicates", n, n,
        f"{dup:,} rows ({dup / n:.1%}) repeat all 11 columns; largest identical group={biggest:,}; "
        "NOT removed (no vehicle ID, each row = one vehicle)")
    return df


# =============================================================================================
# STEP 10 - VALIDATION AGAINST THE REPORT
# =============================================================================================
def validate_against_report(df: pd.DataFrame, strict: bool = False) -> None:
    """
    WHAT: Compare key counts of the processed data with the figures printed in the report.
    HOW : Recompute 11 indicators and compare with REPORT_FIGURES; print OK / DIFF.
    WHY : Guarantees that preprocessing did not silently drop/alter rows and that the processed
          file still reproduces the report's tables (2.2, 2.3, 2.4). With --strict a mismatch
          stops the pipeline (useful in CI).
    """
    actual = {
        "rows (N)": len(df),
        "BEV rows": int((df["Type"] == "BEV").sum()),
        "PHEV rows": int((df["Type"] == "PHEV").sum()),
        "distinct County": df[COUNTY].nunique(),
        "distinct Make": df[MAKE].nunique(),
        "distinct Model (non-null)": df[MODEL].nunique(),
        "distinct Model Year": df[YEAR].nunique(),
        "Electric Range missing": int(df[RANGE].isna().sum()),
        "Base MSRP valid": int(df[MSRP].notna().sum()),
        "Model missing": int(df[MODEL].isna().sum()),
        "CAFV 'Unknown' rows": int((df["CAFV Status"] == "Unknown").sum()),
    }
    diffs = 0
    for k, expected in REPORT_FIGURES.items():
        ok = actual[k] == expected
        diffs += not ok
        print(f"   {'OK  ' if ok else 'DIFF'} {k:28s} processed={actual[k]:>9,} report={expected:>9,}")
    n_pairs = df[["County", "City"]].drop_duplicates().shape[0]
    print(f"   INFO distinct City (after unification) = {df[CITY].nunique()} "
          f"(report quotes 454 raw spellings) | (County, City) pairs = {n_pairs}")
    log("10 validate", len(df), len(df), f"{len(REPORT_FIGURES) - diffs}/{len(REPORT_FIGURES)} report figures reproduced")
    if strict and diffs:
        raise AssertionError(f"{diffs} indicator(s) differ from the report")


# =============================================================================================
# STEP 11 - SAVE
# =============================================================================================
def save(df: pd.DataFrame, out_path: Path) -> None:
    """
    WHAT: Write the processed dataset and the audit trail to disk.
    HOW : to_csv(utf-8-sig) for both files; parent folders are created automatically.
    WHY : Downstream scripts/notebooks (EDA, hypothesis tests) should read ONE reproducible,
          versioned output instead of re-cleaning; the audit CSV is the evidence for report
          section 3.2 (rows before/after each step). utf-8-sig keeps the file Excel-friendly.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(AUDIT).to_csv(AUDIT_OUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df):,} rows x {df.shape[1]} columns -> {out_path}")
    print(f"Saved audit trail -> {AUDIT_OUT}")


# =============================================================================================
# ORCHESTRATION
# =============================================================================================
def run_pipeline(in_path: Path = DEFAULT_IN, out_path: Path = DEFAULT_OUT, strict: bool = False) -> pd.DataFrame:
    """Chain all steps in order. Order matters: validate -> scope -> text -> types -> values
    -> features -> audits -> checks -> save (see docstrings for the dependency of each step)."""
    AUDIT.clear()
    df = load_raw(in_path)
    df = validate_schema(df)
    df = filter_scope(df)
    df = standardize_text(df)
    df = cast_types(df)
    df = handle_invalid_values(df)
    df = engineer_features(df)
    df = audit_missing(df)
    df = audit_duplicates(df)
    print("\nValidation against report figures:")
    validate_against_report(df, strict=strict)
    save(df, out_path)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--strict", action="store_true", help="fail if a figure differs from the report")
    a = ap.parse_args()
    run_pipeline(a.input, a.output, a.strict)
