"""Reconstruct, validate and prepare Chapter 3 data; never edit source/target CSVs."""
from pathlib import Path
import argparse
import json
import platform
import pandas as pd
from clean_population import clean_population, write_csv
from clean_history import clean_history
from verify_preprocessing import compare_csv, sha256

BASE = Path('D:/Phân tích kinh doanh_2627')
REPO = Path(__file__).resolve().parents[2]


def require(condition: bool, description: str) -> None:
    if not condition:
        raise ValueError(description)


def prepare_monthly(history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dates = pd.to_datetime(history.Date, format='%Y-%m-%d', errors='raise')
    count_cols = list(history.columns[4:9])
    require(not history.isna().any().any(), 'History contains missing values.')
    require(not history.duplicated(['Date', 'County', 'Vehicle Primary Use']).any(), 'Duplicate History key.')
    require(set(history.State) == {'WA'}, 'History contains other states.')
    require(dates.dt.is_month_end.all(), 'History contains dates that are not month ends.')
    require(history[count_cols].ge(0).all().all(), 'Negative vehicle count.')
    require(history[count_cols].mod(1).eq(0).all().all(), 'Non-integral vehicle count.')
    require(history.iloc[:, 4].add(history.iloc[:, 5]).eq(history.iloc[:, 6]).all(), 'BEV + PHEV mismatch.')
    require(history.iloc[:, 6].add(history.iloc[:, 7]).eq(history.iloc[:, 8]).all(), 'Total vehicles mismatch.')
    require(history['Total Vehicles'].gt(0).all(), 'Undefined EV percentage denominator.')
    pct = (100 * history['Electric Vehicle (EV) Total'] / history['Total Vehicles']).round(2)
    require(pct.eq(history['Percent Electric Vehicles']).all(), 'Source EV percentage mismatch.')
    expected_keys = pd.MultiIndex.from_product([
        sorted(history.Date.unique()), sorted(history.County.unique()),
        sorted(history['Vehicle Primary Use'].unique()),
    ], names=['Date', 'County', 'Vehicle Primary Use'])
    observed_keys = pd.MultiIndex.from_frame(history[['Date', 'County', 'Vehicle Primary Use']])
    require(not len(expected_keys.difference(observed_keys)), 'Missing month/county/use combinations.')
    monthly = history.groupby('Date', sort=True)[count_cols].sum().reset_index()
    monthly.columns = ['date', 'bev', 'phev', 'ev_total', 'non_ev', 'total_vehicles']
    month_dates = pd.to_datetime(monthly.date)
    require(pd.PeriodIndex(month_dates, freq='M').equals(pd.period_range(month_dates.min(), month_dates.max(), freq='M')), 'Missing months.')
    monthly['ev_pct'] = 100 * monthly.ev_total / monthly.total_vehicles
    monthly['split'] = monthly.date.map(lambda date: 'train' if date <= '2022-11-30' else 'test')
    checks = {
        'missing_values': 0, 'duplicate_keys': 0, 'complete_cartesian_grid': True,
        'count_identities_all_rows': True, 'source_percentages_match_2_decimals': True,
        'consecutive_month_ends': True, 'monthly_shape': list(monthly.shape),
        'split_counts': monthly.split.value_counts().to_dict(),
        'first_month': monthly.iloc[0].to_dict(), 'last_train': monthly[monthly.split.eq('train')].iloc[-1].to_dict(),
        'last_month': monthly.iloc[-1].to_dict(),
    }
    return monthly, checks


def population_profile(population: pd.DataFrame) -> dict:
    groups = {}
    for label, lo, hi in [('1997-2020', 1997, 2020), ('2021-2024', 2021, 2024)]:
        values = population.loc[population['Model Year'].between(lo, hi), 'Electric Range']
        groups[label] = {'total': len(values), 'valid': int(values.notna().sum()), 'missing': int(values.isna().sum()), 'missing_pct': float(values.isna().mean() * 100)}
    unknown = population['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].eq('Eligibility unknown as battery range has not been researched')
    return {
        'distinct_cities': int(population.City.nunique()),
        'distinct_counties': int(population.County.nunique()),
        'vehicle_types': population['Electric Vehicle Type'].value_counts().to_dict(),
        'valid_range_H1_H2': int(population['Electric Range'].notna().sum()),
        'known_CAFV': int((~unknown).sum()), 'unknown_CAFV': int(unknown.sum()),
        'range_by_year_group': groups,
        'missing_model_kept_as_missing': int(population.Model.isna().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population-raw', type=Path, default=BASE/'archive/Electric_Vehicle_Population_Data.csv')
    parser.add_argument('--history-raw', type=Path, default=BASE/'Electric_Vehicle_Population_Size_History_By_County.csv/Electric_Vehicle_Population_Size_History_By_County.csv')
    parser.add_argument('--population-reference', type=Path, default=BASE/'archive/EV_Population_Washington_Clean (1).csv')
    parser.add_argument('--history-reference', type=Path, default=BASE/'Electric_Vehicle_Population_Size_History_By_County.csv/EV_Population_History_Washington_Clean.csv')
    parser.add_argument('--output-dir', type=Path, default=REPO/'data/reconstructed_chapter3')
    args = parser.parse_args()
    outputs = {key: args.output_dir / name for key, name in {
        'population': 'EV_Population_Washington_Clean.csv',
        'history': 'EV_Population_History_Washington_Clean.csv',
        'monthly': 'History_ToanBang_TheoThang.csv', 'audit': 'verification.json',
    }.items()}
    inputs = [args.population_raw, args.history_raw, args.population_reference, args.history_reference]
    for path in inputs:
        require(path.is_file(), f'Input not found: {path}')
    for path in outputs.values():
        require(path.resolve() not in [p.resolve() for p in inputs], 'Output would overwrite an input.')
        require(not path.exists(), f'Output already exists; choose a new --output-dir: {path}')
    source_hashes = {str(path.resolve()): sha256(path) for path in inputs}
    # Only raw paths enter these transformation functions.
    population, population_audit = clean_population(args.population_raw)
    history, history_audit = clean_history(args.history_raw)
    write_csv(population, outputs['population'])
    write_csv(history, outputs['history'])
    verification = {
        'population': compare_csv(outputs['population'], args.population_reference),
        'history': compare_csv(outputs['history'], args.history_reference),
    }
    monthly, monthly_checks = prepare_monthly(history)
    write_csv(monthly, outputs['monthly'])
    require(monthly_checks['split_counts'] == {'train': 71, 'test': 12}, 'Unexpected train/test counts.')
    for path in inputs:
        require(sha256(path) == source_hashes[str(path.resolve())], f'Source changed: {path}')
    report = {
        'runtime': {'python': platform.python_version(), 'pandas': pd.__version__},
        'all_four_inputs_unchanged': True, 'input_sha256': source_hashes,
        'population_process': population_audit, 'history_process': history_audit,
        'verification': verification, 'population_profile': population_profile(population),
        'monthly_checks': monthly_checks,
        'scope': 'Reconstructed equivalent pipeline, not recovered original source code. No imputation, deduplication, label replacement or outlier removal.',
    }
    outputs['audit'].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'verification': verification, 'monthly_checks': monthly_checks, 'audit': str(outputs['audit'])}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
