#!/usr/bin/env python3
"""Zip raw WAV, scripts, and Spark runners for DGX Spark."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import common

_SKIP_NAMES = {
    'probe_en_45s.wav',
    'probe_ch3_720_750.wav',
    'vertin_cosyvoice3_spark.zip',
}


def add_tree(
    archive: zipfile.ZipFile,
    root: Path,
    prefix: str,
) -> int:
    """Add a directory tree to the zip. Returns the number of files added."""
    count = 0
    if not root.exists():
        return count
    for path in root.rglob('*'):
        if not path.is_file() or path.name in _SKIP_NAMES:
            continue
        archive.write(path, f'{prefix}/{path.relative_to(root).as_posix()}')
        count += 1
    return count


def pack(dest: Path) -> Path:
    """Create a transfer zip that Spark can unzip and run."""
    common.ensure_dir(dest.parent)
    with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        n_raw = add_tree(archive, common.RAW_WAV_DIR, 'data/raw_wav')
        n_scripts = add_tree(archive, common.PROJECT_ROOT / 'scripts', 'scripts')
        n_spark = add_tree(archive, common.PROJECT_ROOT / 'spark', 'spark')
        req = common.PROJECT_ROOT / 'requirements-dataprep.txt'
        if req.exists():
            archive.write(req, 'requirements-dataprep.txt')
    print(
        f'Wrote {dest} (raw_wav={n_raw}, scripts={n_scripts}, spark={n_spark})'
    )
    print('On Spark:')
    print('  unzip vertin_spark_full.zip -d ~/voice_maker')
    print('  cd ~/voice_maker && bash spark/run_all.sh')
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out',
        type=Path,
        default=common.PROCESSED_DIR / 'vertin_spark_full.zip',
    )
    args = parser.parse_args()
    pack(args.out)


if __name__ == '__main__':
    main()
