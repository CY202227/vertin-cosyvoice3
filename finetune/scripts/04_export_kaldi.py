#!/usr/bin/env python3
"""Export CosyVoice3 Kaldi-style lists from the processed manifest."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import common


def write_kaldi_split(split_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write wav.scp/text/utt2spk/spk2utt/instruct for one split."""
    common.ensure_dir(split_dir)
    rows = sorted(rows, key=lambda item: str(item['utt_id']))
    speaker_to_utts: dict[str, list[str]] = defaultdict(list)

    with (split_dir / 'wav.scp').open('w', encoding='utf-8', newline='\n') as wav_scp, (
        split_dir / 'text'
    ).open('w', encoding='utf-8', newline='\n') as text_f, (
        split_dir / 'utt2spk'
    ).open('w', encoding='utf-8', newline='\n') as utt2spk, (
        split_dir / 'instruct'
    ).open('w', encoding='utf-8', newline='\n') as instruct:
        for row in rows:
            utt_id = str(row['utt_id'])
            wav_path = common.PROJECT_ROOT / str(row['clip_wav'])
            transcript = str(row['text']).strip()
            if not transcript:
                transcript = '[vocalized-noise]'
            wav_scp.write(f'{utt_id} {wav_path.as_posix()}\n')
            text_f.write(f'{utt_id} {transcript}\n')
            utt2spk.write(f'{utt_id} {common.SPEAKER_ID}\n')
            instruct.write(f'{utt_id} {common.INSTRUCT_TEXT}\n')
            speaker_to_utts[common.SPEAKER_ID].append(utt_id)

    with (split_dir / 'spk2utt').open('w', encoding='utf-8', newline='\n') as spk2utt:
        for speaker, utts in speaker_to_utts.items():
            spk2utt.write(f'{speaker} {" ".join(utts)}\n')

    (split_dir / 'wav.scp.rel').write_text(
        ''.join(
            f'{row["utt_id"]} {row["clip_wav"]}\n'
            for row in rows
        ),
        encoding='utf-8',
    )


def split_rows(
    rows: list[dict[str, Any]],
    dev_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by source file so each chapter contributes to train and dev."""
    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[int(row['source_index'])].append(row)

    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for source_index in sorted(by_source):
        group = list(by_source[source_index])
        rng.shuffle(group)
        if len(group) == 1:
            train.extend(group)
            continue
        n_dev = max(1, int(round(len(group) * dev_ratio)))
        n_dev = min(n_dev, len(group) - 1)
        dev.extend(group[:n_dev])
        train.extend(group[n_dev:])
    return train, dev


def export_kaldi(dev_ratio: float, seed: int) -> None:
    """Write train/dev Kaldi directories from all.jsonl."""
    if not common.ALL_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f'{common.ALL_MANIFEST_PATH} is missing. Run 03_attach_nonspeech.py first.'
        )
    rows = [
        row
        for row in common.read_jsonl(common.ALL_MANIFEST_PATH)
        if str(row.get('text', '')).strip()
    ]
    if not rows:
        raise RuntimeError('Manifest has no non-empty transcripts.')

    train_rows, dev_rows = split_rows(rows, dev_ratio=dev_ratio, seed=seed)
    write_kaldi_split(common.KALDI_DIR / 'train', train_rows)
    write_kaldi_split(common.KALDI_DIR / 'dev', dev_rows)
    print(
        f'Exported {len(train_rows)} train / {len(dev_rows)} dev utterances '
        f'to {common.KALDI_DIR}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1986)
    args = parser.parse_args()
    export_kaldi(args.dev_ratio, args.seed)


if __name__ == '__main__':
    main()
