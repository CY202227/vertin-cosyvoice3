#!/usr/bin/env python3
"""Run the local Vertin data-prep stages in order."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import importlib

import common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--from-stage',
        type=int,
        default=1,
        choices=(1, 2, 3, 4),
        help='First stage to run (1 extract, 2 asr, 3 nonspeech, 4 kaldi).',
    )
    parser.add_argument(
        '--backend',
        choices=('funasr-nano', 'funasr', 'whisper'),
        default='funasr-nano',
    )
    parser.add_argument('--whisper-model', default='distil-large-v3')
    parser.add_argument('--language', default='en', choices=('en', 'zh', 'auto'))
    args = parser.parse_args()

    if args.from_stage <= 1:
        extract = importlib.import_module('01_extract_audio')
        extract.extract_all(common.DATA_DIR, common.RAW_WAV_DIR)
    if args.from_stage <= 2:
        asr_cut = importlib.import_module('02_asr_cut')
        asr_cut.run_asr_cut(
            args.backend, args.whisper_model, language=args.language
        )
    if args.from_stage <= 3:
        attach = importlib.import_module('03_attach_nonspeech')
        attach.attach_nonspeech()
    if args.from_stage <= 4:
        export = importlib.import_module('04_export_kaldi')
        export.export_kaldi(dev_ratio=0.05, seed=1986)


if __name__ == '__main__':
    main()
