#!/usr/bin/env python3
"""Download Fun-CosyVoice3-0.5B weights into runtime/pretrained_models."""

from pathlib import Path

from modelscope import snapshot_download

DEST = Path(__file__).resolve().parent / 'pretrained_models' / 'Fun-CosyVoice3-0.5B'
DEST.mkdir(parents=True, exist_ok=True)
print('downloading to', DEST)
snapshot_download(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir=str(DEST),
)
print('done', DEST)
