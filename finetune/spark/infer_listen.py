#!/usr/bin/env python3
"""Synthesize jsonl lines 35-58 with the best SFT LLM checkpoint."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio

_FINETUNE = Path(__file__).resolve().parents[1]
_REPO = _FINETUNE.parent
ROOT = Path(os.environ.get('SPARK_WORKSPACE', str(_FINETUNE)))
COSYVOICE_DIR = Path(
    os.environ.get('COSYVOICE_DIR', _REPO / 'runtime' / 'CosyVoice')
)
MODEL_DIR = Path(
    os.environ.get(
        'MODEL_DIR',
        _REPO / 'runtime' / 'pretrained_models' / 'Fun-CosyVoice3-0.5B',
    )
)
CKPT = Path(
    os.environ.get('LLM_CKPT', _FINETUNE / 'ckpts' / 'epoch_8_whole.pt')
)
MANIFEST = ROOT / 'data/processed/manifests/all.jsonl'
OUT_DIR = ROOT / 'export' / 'listen_epoch8'
LINE_START = 35
LINE_END = 58

sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / 'third_party' / 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402


_REPLACEMENTS = (
    (r'\[vocal\]', ''),
    (r'\[breath\]', ''),
    (r'\[sigh\]', ''),
    (r'は\s*い\s*。?', ''),
    (r'\bregul\s+us\b', 'Regulus'),
    (r'\bReg\s+ulus\b', 'Regulus'),
    (r'\breg\s+ulars\b', 'Regulus'),
    (r'\bar\s+can\s+ists\b', 'arcanists'),
    (r'\bSon\s+net\s+to\b', 'Sonetto'),
    (r'\bSon\s+net\s+o\b', 'Sonetto'),
    (r'\bSon\s+eto\b', 'Sonetto'),
    (r'\bson\s+net\s+to\b', 'Sonetto'),
    (r'\bson\s+ato\b', 'Sonetto'),
    (r'\bC\s+en\s+eto\b', 'Sonetto'),
    (r'\bPav\s+lov\b', 'Pavlov'),
    (r'\bMan\s+us\s+ville\b', 'Manusville'),
    (r'\binjust\s+ices\b', 'injustices'),
    (r'\borphan\s+ages\b', 'orphanages'),
    (r'\bad\s+mon\s+ition\b', 'admonition'),
    (r'\bU\s+nt\s+il\b', 'Until'),
    (r'\bdep\s+rive\b', 'deprive'),
    (r'\btwent\s+ieth\b', 'twentieth'),
    (r'\bSpe\s+aking\b', 'Speaking'),
    (r'\bH\s+ide\b', 'Hide'),
    (r'\bEn\s+ough\b', 'Enough'),
    (r'\bDam\s+n\s+it\b', 'Damn it'),
    (r'\bho\s+ly\s+potion\b', 'holy potion'),
    (r'\bTHE\s+WOO\s+DEN\b', 'the wooden'),
    (r'\bward\s+en\b', 'warden'),
    (r'\bmen\s+ace\b', 'Manus'),
    (r'\bm\s+r\s+apple\b', 'Mr Apple'),
    (r'\bMr\s+Apple\b', 'Mr Apple'),
    (r'\bm\s+ine\b', 'mine'),
    (r'\bUp\s+date\b', 'Update'),
    (r'\bselect\s+s\b', 'selects'),
    (r'\bAl\s+so\b', 'Also'),
)


def clean_text(text: str) -> str:
    """Fix the worst ASR token splits so TTS is listen-able."""
    out = text
    for pattern, repl in _REPLACEMENTS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    out = re.sub(r'\s+', ' ', out)
    out = re.sub(r'\s+([,.!?])', r'\1', out)
    return out.strip(' ,.')


def load_sft_llm(cosyvoice: CosyVoice3, ckpt: Path) -> None:
    """Replace the base LLM with a torch_ddp SFT checkpoint."""
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    state.pop('epoch', None)
    state.pop('step', None)
    missing, unexpected = cosyvoice.model.llm.load_state_dict(state, strict=False)
    print('loaded', ckpt, 'missing', len(missing), 'unexpected', len(unexpected))
    # SFT ckpt is almost all bf16; CosyVoice3 inference without fp16
    # autocast feeds float32 activations into those weights.
    cosyvoice.model.llm.float()
    cosyvoice.model.llm.to(cosyvoice.model.device).eval()


def concat_wavs(paths: list[Path], out_path: Path, gap_sec: float = 0.6) -> None:
    waves = []
    sr = None
    for path in paths:
        audio, this_sr = soundfile.read(str(path), always_2d=True, dtype='float32')
        if sr is None:
            sr = this_sr
        elif this_sr != sr:
            wav = torch.from_numpy(audio.T)
            wav = torchaudio.functional.resample(wav, this_sr, sr)
            audio = wav.numpy().T
        waves.append(audio)
        waves.append(np.zeros((int(sr * gap_sec), audio.shape[1]), dtype=np.float32))
    soundfile.write(str(out_path), np.concatenate(waves, axis=0), sr)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    synth_dir = OUT_DIR / 'synth'
    ref_dir = OUT_DIR / 'ref'
    synth_dir.mkdir(exist_ok=True)
    ref_dir.mkdir(exist_ok=True)

    lines = MANIFEST.read_text(encoding='utf-8').splitlines()
    items = [json.loads(lines[i - 1]) for i in range(LINE_START, LINE_END + 1)]
    prompt = json.loads(lines[33])  # vertin_01_00034, not in 35-58
    prompt_wav = str(ROOT / prompt['clip_wav'])
    prompt_text = (
        'You are a helpful assistant.<|endofprompt|>'
        + clean_text(prompt['text'])
    )
    print('prompt', prompt['utt_id'], prompt_text)

    print('loading CosyVoice3 from', MODEL_DIR)
    cosyvoice = CosyVoice3(str(MODEL_DIR), load_trt=False, fp16=False)
    load_sft_llm(cosyvoice, CKPT)

    synth_paths = []
    texts_path = OUT_DIR / 'texts.tsv'
    with texts_path.open('w', encoding='utf-8') as fout:
        for item in items:
            utt_id = item['utt_id']
            text = clean_text(item['text'])
            ref_src = ROOT / item['clip_wav']
            ref_dst = ref_dir / f'{utt_id}.wav'
            if ref_src.exists() and not ref_dst.exists():
                ref_dst.write_bytes(ref_src.read_bytes())
            out_wav = synth_dir / f'{utt_id}.wav'
            fout.write(f'{utt_id}\t{text}\n')
            print('===', utt_id, text[:80])
            chunks = []
            try:
                for output in cosyvoice.inference_zero_shot(
                    text,
                    prompt_text,
                    prompt_wav,
                    stream=False,
                    text_frontend=True,
                ):
                    chunks.append(output['tts_speech'].cpu())
            except Exception as exc:
                print('FAILED', utt_id, type(exc).__name__, exc)
                continue
            if not chunks:
                print('empty synth', utt_id)
                continue
            speech = torch.cat(chunks, dim=1).squeeze(0).numpy()
            soundfile.write(str(out_wav), speech, cosyvoice.sample_rate)
            synth_paths.append(out_wav)

    if synth_paths:
        concat_wavs(synth_paths, OUT_DIR / 'all_synth.wav', gap_sec=0.8)

    # Short A/B: original then synth for a few cleaner lines.
    ab_ids = {
        'vertin_02_00040',
        'vertin_02_00041',
        'vertin_02_00046',
        'vertin_02_00047',
        'vertin_02_00049',
        'vertin_02_00052',
        'vertin_02_00058',
    }
    ab_paths = []
    for item in items:
        utt_id = item['utt_id']
        if utt_id not in ab_ids:
            continue
        ref = ref_dir / f'{utt_id}.wav'
        syn = synth_dir / f'{utt_id}.wav'
        if ref.exists() and syn.exists():
            ab_paths.extend([ref, syn])
    if ab_paths:
        concat_wavs(ab_paths, OUT_DIR / 'ab_ref_then_synth.wav', gap_sec=0.9)
    print('wrote', OUT_DIR)


if __name__ == '__main__':
    main()
