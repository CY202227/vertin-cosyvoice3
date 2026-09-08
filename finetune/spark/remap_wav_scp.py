#!/usr/bin/env python3
"""Rewrite Kaldi wav.scp paths after copying clips onto DGX Spark."""

from __future__ import annotations

import argparse
from pathlib import Path


def remap(rel_scp: Path, dest_scp: Path, clips_root: Path) -> None:
    """Turn repo-relative clip paths into absolute Spark paths."""
    lines: list[str] = []
    with rel_scp.open('r', encoding='utf-8') as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            utt_id, rel_path = raw.split(maxsplit=1)
            # rel_path looks like data/processed/clips/vertin/utt.wav
            name = Path(rel_path).name
            abs_path = (clips_root / name).resolve()
            lines.append(f'{utt_id} {abs_path.as_posix()}\n')
    dest_scp.write_text(''.join(lines), encoding='utf-8')
    print(f'Wrote {dest_scp} ({len(lines)} utterances)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rel-scp', type=Path, required=True)
    parser.add_argument('--dest-scp', type=Path, required=True)
    parser.add_argument(
        '--clips-root',
        type=Path,
        required=True,
        help='Directory that contains the copied vertin_*.wav clips.',
    )
    args = parser.parse_args()
    remap(args.rel_scp, args.dest_scp, args.clips_root)


if __name__ == '__main__':
    main()
