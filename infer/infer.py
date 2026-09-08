#!/usr/bin/env python3
"""Synthesize Vertin speech from text with the CosyVoice3 SFT LLM.

Examples:
  python infer/infer.py "Hi, Sonetto."
  python infer/infer.py -f infer/examples/story.txt -o infer/outputs/story.wav
  python infer/infer.py --speed 1.1 "Slightly faster."
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch

INFER_DIR = Path(__file__).resolve().parent
REPO = INFER_DIR.parent
COSYVOICE_DIR = Path(os.environ.get('COSYVOICE_DIR', REPO / 'runtime' / 'CosyVoice'))
MODEL_DIR = Path(
    os.environ.get(
        'MODEL_DIR',
        REPO / 'runtime' / 'pretrained_models' / 'Fun-CosyVoice3-0.5B',
    )
)
CKPT = Path(
    os.environ.get(
        'LLM_CKPT',
        REPO / 'finetune' / 'ckpts' / 'epoch_8_whole.pt',
    )
)
ASSETS = INFER_DIR / 'assets'
DEFAULT_PROMPT_WAV = Path(os.environ.get('PROMPT_WAV', ASSETS / 'prompt.wav'))
DEFAULT_PROMPT_TXT = Path(os.environ.get('PROMPT_TXT', ASSETS / 'prompt.txt'))
INSTRUCT = 'You are a helpful assistant.<|endofprompt|>'

sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / 'third_party' / 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402


def load_prompt_transcript(path: Path) -> str:
    env_text = os.environ.get('PROMPT_TEXT')
    if env_text:
        return env_text.strip()
    if not path.exists():
        raise SystemExit(f'missing prompt transcript: {path}')
    return path.read_text(encoding='utf-8').strip()


def chunk_text(text: str, prompt_text: str) -> list[str]:
    """Split long text so CosyVoice is not starved vs the prompt."""
    paras = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', text)]
    paras = [p for p in paras if p]
    if not paras:
        paras = [re.sub(r'\s+', ' ', text).strip()]
    min_len = max(90, int(0.55 * len(prompt_text)))
    chunks: list[str] = []
    buf = ''
    for para in paras:
        candidate = f'{buf} {para}'.strip() if buf else para
        if buf and len(candidate) > 320:
            chunks.append(buf)
            buf = para
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_len:
            merged[-1] = f'{merged[-1]} {chunk}'.strip()
        else:
            merged.append(chunk)
    return [c for c in merged if c]


def load_sft_llm(cosyvoice: CosyVoice3, ckpt: Path) -> None:
    """Replace the base LLM with the Vertin SFT checkpoint."""
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    state.pop('epoch', None)
    state.pop('step', None)
    missing, unexpected = cosyvoice.model.llm.load_state_dict(
        state, strict=False
    )
    print('loaded', ckpt.name, 'missing', len(missing), 'unexpected', len(unexpected))
    cosyvoice.model.llm.float()
    cosyvoice.model.llm.to(cosyvoice.model.device).eval()


def concat_wavs(waves: list[np.ndarray], sr: int, gap_sec: float) -> np.ndarray:
    parts = []
    gap = np.zeros(int(sr * gap_sec), dtype=np.float32)
    for i, wav in enumerate(waves):
        parts.append(np.asarray(wav, dtype=np.float32).reshape(-1))
        if i != len(waves) - 1:
            parts.append(gap)
    return np.concatenate(parts, axis=0)


def read_input_text(args: argparse.Namespace) -> str:
    if args.file is not None:
        return args.file.read_text(encoding='utf-8').strip()
    if args.text:
        return ' '.join(args.text).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print('Enter text, then Ctrl+Z and Enter (Windows) or Ctrl+D (Unix):')
    return sys.stdin.read().strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', nargs='*', help='Text to speak')
    parser.add_argument(
        '-f',
        '--file',
        type=Path,
        help='UTF-8 text file to read instead of argv',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=INFER_DIR / 'outputs' / 'tts.wav',
        help='Output wav path',
    )
    parser.add_argument('--prompt-wav', type=Path, default=DEFAULT_PROMPT_WAV)
    parser.add_argument('--prompt-txt', type=Path, default=DEFAULT_PROMPT_TXT)
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='CosyVoice vocoder speed; >1 is faster, <1 is slower',
    )
    parser.add_argument('--fp16', action='store_true', help='Enable CosyVoice fp16')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = read_input_text(args)
    if not text:
        raise SystemExit('empty text')
    if not args.prompt_wav.exists():
        raise SystemExit(f'missing prompt wav: {args.prompt_wav}')

    prompt_transcript = load_prompt_transcript(args.prompt_txt)
    prompt_text = INSTRUCT + prompt_transcript
    chunks = chunk_text(text, prompt_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print('prompt wav', args.prompt_wav)
    print('prompt asr', prompt_transcript)
    print('chunks', len(chunks), 'cosy speed', args.speed)
    print('loading CosyVoice3 (wetext 403 is OK, wait for GPU load)')

    cosyvoice = CosyVoice3(str(MODEL_DIR), load_trt=False, fp16=args.fp16)
    load_sft_llm(cosyvoice, CKPT)

    waves = []
    for i, chunk in enumerate(chunks, 1):
        print(f'=== {i}/{len(chunks)} ({len(chunk)} chars) {chunk[:80]}')
        pieces = []
        for output in cosyvoice.inference_zero_shot(
            chunk,
            prompt_text,
            str(args.prompt_wav),
            stream=False,
            speed=args.speed,
            text_frontend=True,
        ):
            pieces.append(output['tts_speech'].cpu())
        if not pieces:
            raise SystemExit(f'empty synth for chunk {i}')
        speech = torch.cat(pieces, dim=1).squeeze(0).numpy()
        waves.append(speech)
        print('chunk sec', round(len(speech) / cosyvoice.sample_rate, 2))

    audio = concat_wavs(waves, cosyvoice.sample_rate, gap_sec=0.4)
    soundfile.write(str(args.output), audio, cosyvoice.sample_rate)
    print(
        'wrote',
        args.output,
        'sec',
        round(len(audio) / cosyvoice.sample_rate, 2),
    )


if __name__ == '__main__':
    main()
