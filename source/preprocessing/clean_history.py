"""Reconstruct county/use/month History from raw alone, without aggregation."""
from pathlib import Path
import argparse
import pandas as pd
from clean_population import write_csv

COLUMNS = [
    'Date', 'County', 'State', 'Vehicle Primary Use',
    'Battery Electric Vehicles (BEVs)', 'Plug-In Hybrid Electric Vehicles (PHEVs)',
    'Electric Vehicle (EV) Total', 'Non-Electric Vehicle Total',
    'Total Vehicles', 'Percent Electric Vehicles',
]


def clean_history(raw_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(raw_path, low_memory=False)
    if list(raw.columns) != COLUMNS:
        raise ValueError('History source schema differs from the supplied source.')
    clean = raw.loc[raw['State'].eq('WA'), COLUMNS].copy()
    clean['Date'] = pd.to_datetime(clean['Date'], format='%B %d %Y', errors='raise').dt.strftime('%Y-%m-%d')
    for col in COLUMNS[4:]:
        clean[col] = pd.to_numeric(clean[col], errors='raise').astype('float64')
    # Do not sort Vehicle Primary Use: the target preserves source order in ties.
    clean = clean.sort_values(['Date', 'County'], kind='stable').reset_index(drop=True)
    audit = {
        'raw_shape': list(raw.shape), 'clean_shape': list(clean.shape),
        'removed_rows_non_WA_or_missing_state': int(len(raw) - len(clean)),
        'raw_missing_state': int(raw['State'].isna().sum()),
        'missing_after': {col: int(n) for col, n in clean.isna().sum().items()},
        'dates': int(clean.Date.nunique()), 'counties': int(clean.County.nunique()),
        'uses': clean['Vehicle Primary Use'].value_counts().to_dict(),
        'first_date': clean.Date.min(), 'last_date': clean.Date.max(),
        'sort_keys': ['Date', 'County'], 'sort_ties': 'preserve source order',
    }
    return clean, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.raw.resolve() or args.output.exists():
        parser.error('Output must be a new path, distinct from the source.')
    clean, audit = clean_history(args.raw)
    write_csv(clean, args.output)
    print(audit)


if __name__ == '__main__':
    main()
