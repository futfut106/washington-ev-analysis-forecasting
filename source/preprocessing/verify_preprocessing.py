"""Independent target comparison; clean scripts never read the target files."""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compare_csv(generated: Path, reference: Path) -> dict:
    actual = pd.read_csv(generated, low_memory=False)
    expected = pd.read_csv(reference, low_memory=False)
    # Checks column order, row order, missing positions and every cell value.
    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_dtype=True)
    actual_hash, expected_hash = sha256(generated), sha256(reference)
    if actual_hash != expected_hash:
        raise AssertionError(f'Values match but file bytes differ: {generated.name}')
    return {
        'generated': str(generated.resolve()), 'reference': str(reference.resolve()),
        'shape': list(actual.shape), 'all_cells_and_row_order_equal': True,
        'byte_identical': True, 'generated_sha256': actual_hash,
        'reference_sha256': expected_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_csv(args.generated, args.reference), indent=2))


if __name__ == '__main__':
    main()
