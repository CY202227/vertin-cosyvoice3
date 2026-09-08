#!/usr/bin/env python3
"""Export epoch_8 SFT LLM into a drop-in CosyVoice3 model directory."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / 'runtime' / 'pretrained_models' / 'Fun-CosyVoice3-0.5B'
DEST = REPO / 'runtime' / 'pretrained_models' / 'Vertin-CosyVoice3-0.5B'
CKPT = REPO / 'finetune' / 'ckpts' / 'epoch_8_whole.pt'


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink files / junction directories; fall back to copy."""
    if dst.exists() or dst.is_symlink():
        return
    try:
        if src.is_dir():
            os.symlink(src, dst, target_is_directory=True)
        else:
            os.link(src, dst)
        print('link', dst.name)
        return
    except OSError as error:
        print('link failed', dst.name, error)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print('copy', dst.name)


def main() -> None:
    if not BASE.exists():
        raise SystemExit(f'missing base model: {BASE}')
    if not CKPT.exists():
        raise SystemExit(f'missing SFT ckpt: {CKPT}')
    DEST.mkdir(parents=True, exist_ok=True)

    for item in BASE.iterdir():
        if item.name == 'llm.pt':
            continue
        link_or_copy(item, DEST / item.name)

    print('loading', CKPT)
    state = torch.load(CKPT, map_location='cpu', weights_only=False)
    state.pop('epoch', None)
    state.pop('step', None)
    # Official llm.pt is float32; SFT tensors are mostly bf16.
    state = {key: tensor.float().contiguous() for key, tensor in state.items()}
    llm_path = DEST / 'llm.pt'
    print('writing', llm_path)
    torch.save(state, llm_path)
    readme = DEST / 'VERTIN.md'
    readme.write_text(
        'Vertin CosyVoice3: official flow/hift + epoch_8 SFT LLM.\n'
        'Load with CosyVoice3(this_dir). Do not overlay another llm.pt.\n',
        encoding='utf-8',
    )
    print('merged model', DEST)
    print('llm.pt MB', round(llm_path.stat().st_size / 1024 / 1024, 1))


if __name__ == '__main__':
    main()
