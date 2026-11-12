"""Reconstruct Population from raw alone; preserve vehicle multiplicities."""
from pathlib import Path
import argparse
import pandas as pd

COLUMNS = [
    'County', 'City', 'State', 'Postal Code', 'Model Year', 'Make', 'Model',
    'Electric Vehicle Type', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility',
    'Electric Range', 'Base MSRP',
]
SORT_KEYS = ['County', 'City', 'Model Year', 'Make']
NUMERIC = ['Postal Code', 'Model Year', 'Electric Range', 'Base MSRP']


def clean_population(raw_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(raw_path, low_memory=False)
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing source columns: {sorted(missing)}')
    clean = raw.loc[raw['State'].eq('WA'), COLUMNS].copy()
    zeros = {col: int(clean[col].eq(0).sum()) for col in ['Electric Range', 'Base MSRP']}
    for col in zeros:
        clean[col] = clean[col].mask(clean[col].eq(0))
    # These float representations reproduce the supplied CSV, including year .0.
    # CSV itself has no schema; a later analytical view may use integer years.
    for col in NUMERIC:
        clean[col] = pd.to_numeric(clean[col], errors='raise').astype('float64')
    clean = clean.sort_values(SORT_KEYS, kind='stable', na_position='last').reset_index(drop=True)
    audit = {
        'raw_shape': list(raw.shape), 'clean_shape': list(clean.shape),
        'removed_rows_non_WA': int(len(raw) - len(clean)),
        'removed_columns': [col for col in raw if col not in COLUMNS],
        'zeros_converted_to_missing': zeros,
        'missing_after': {col: int(n) for col, n in clean.isna().sum().items()},
        'duplicates_on_original_all_columns': int(raw.duplicated().sum()),
        'repeated_rows_on_retained_columns_preserved': int(clean.duplicated().sum()),
        'sort_keys': SORT_KEYS, 'sort_ties': 'preserve source order',
    }
    return clean, audit


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write exactly the supplied format: UTF-8 BOM, LF, no index, empty NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig', lineterminator='\n', na_rep='')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.raw.resolve() or args.output.exists():
        parser.error('Output must be a new path, distinct from the source.')
    clean, audit = clean_population(args.raw)
    write_csv(clean, args.output)
    print(audit)


if __name__ == '__main__':
    main()
