#!/usr/bin/env python3
"""Extract 16 kHz mono WAV from every Vertin compilation MP4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import common


def extract_all(data_dir: Path, raw_wav_dir: Path) -> list[dict[str, object]]:
    """Extract every MP4 under data_dir into raw_wav_dir."""
    videos = common.iter_source_videos(data_dir)
    if not videos:
        raise FileNotFoundError(f'No MP4 files found under {data_dir}')

    common.ensure_dir(raw_wav_dir)
    rows: list[dict[str, object]] = []
    for index, video in enumerate(videos, start=1):
        stem = common.sanitize_stem(video.stem, fallback=f'source_{index:02d}')
        wav_name = f'{index:02d}_{stem}.wav'
        dest = raw_wav_dir / wav_name
        print(f'[{index}/{len(videos)}] {video.name} -> {dest.name}')
        common.extract_wav(video, dest)
        rows.append({
            'source_index': index,
            'source_video': common.relative_to_project(video),
            'raw_wav': common.relative_to_project(dest),
            'source_name': video.name,
        })
    common.write_jsonl(common.SOURCE_MAP_PATH, rows)
    print(f'Wrote {common.SOURCE_MAP_PATH} ({len(rows)} files)')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=common.DATA_DIR,
        help='Directory that contains the Vertin MP4 folders.',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=common.RAW_WAV_DIR,
        help='Directory for extracted 16 kHz mono WAV files.',
    )
    args = parser.parse_args()
    extract_all(args.data_dir, args.out_dir)


if __name__ == '__main__':
    main()
