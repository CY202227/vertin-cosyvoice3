#!/usr/bin/env python3
"""Patch CosyVoice sources for DGX Spark torch_ddp / CPU ONNX."""

from __future__ import annotations

import os
from pathlib import Path


def _replace_once(path: Path, old: str, new: str) -> None:
    """Replace old with new if present; skip if already patched."""
    text = path.read_text(encoding='utf-8')
    if new in text:
        print('already patched', path)
        return
    if old not in text:
        raise RuntimeError(f'Patch target not found in {path}')
    path.write_text(text.replace(old, new, 1), encoding='utf-8')
    print('patched', path)


def main() -> None:
    root = Path(os.environ['COSYVOICE_DIR'])

    _replace_once(
        root / 'tools' / 'extract_speech_token.py',
        'providers = ["CUDAExecutionProvider"]',
        (
            'providers = (["CUDAExecutionProvider"] '
            'if "CUDAExecutionProvider" in onnxruntime.get_available_providers() '
            'else ["CPUExecutionProvider"])'
        ),
    )

    _replace_once(
        root / 'cosyvoice' / 'bin' / 'train.py',
        'import deepspeed\n',
        (
            'try:\n'
            '    import deepspeed\n'
            'except ImportError:\n'
            '    deepspeed = None\n'
        ),
    )

    _replace_once(
        root / 'cosyvoice' / 'utils' / 'train_utils.py',
        'import deepspeed\n',
        (
            'try:\n'
            '    import deepspeed\n'
            'except ImportError:\n'
            '    deepspeed = None\n'
        ),
    )
    _replace_once(
        root / 'cosyvoice' / 'utils' / 'train_utils.py',
        (
            'from deepspeed.runtime.zero.stage_1_and_2 import '
            'estimate_zero2_model_states_mem_needs_all_live\n'
        ),
        (
            'try:\n'
            '    from deepspeed.runtime.zero.stage_1_and_2 import (\n'
            '        estimate_zero2_model_states_mem_needs_all_live,\n'
            '    )\n'
            'except ImportError:\n'
            '    estimate_zero2_model_states_mem_needs_all_live = None\n'
        ),
    )
    _replace_once(
        root / 'cosyvoice' / 'bin' / 'train.py',
        'parser = deepspeed.add_config_arguments(parser)\n',
        (
            'if deepspeed is not None:\n'
            '        parser = deepspeed.add_config_arguments(parser)\n'
        ),
    )
    _replace_once(
        root / 'tools' / 'make_parquet_list.py',
        "utt2embedding = torch.load('{}/utt2embedding.pt'.format(args.src_dir)) if os.path.exists('{}/utt2embedding.pt'.format(args.src_dir)) else None\n",
        (
            "def _load_pt(path):\n"
            "        return torch.load(path, map_location='cpu', weights_only=False)\n"
            "\n"
            "    utt2embedding = _load_pt('{}/utt2embedding.pt'.format(args.src_dir)) if os.path.exists('{}/utt2embedding.pt'.format(args.src_dir)) else None\n"
        ),
    )
    _replace_once(
        root / 'cosyvoice' / 'dataset' / 'processor.py',
        "sample['speech'], sample['sample_rate'] = torchaudio.load(BytesIO(sample['audio_data']))\n",
        "sample['speech'], sample['sample_rate'] = load_wav_bytes(sample['audio_data'])\n",
    )
    # PyTorch 2.14 ProcessGroup has no .options; skip join on 1 GPU.
    join_path = root / 'cosyvoice' / 'utils' / 'train_utils.py'
    join_text = join_path.read_text(encoding='utf-8')
    if 'if world_size <= 1 or info_dict["batch_idx"] == 0:' in join_text:
        print('already patched', join_path, 'cosyvoice_join')
    else:
        _replace_once(
            join_path,
            'timeout=group_join.options._timeout)',
            'timeout=datetime.timedelta(seconds=int(info_dict.get("timeout", 60))))',
        )

    # GradScaler cannot unscale bf16 grads on PyTorch 2.14+.
    _replace_once(
        root / 'cosyvoice' / 'bin' / 'train.py',
        'scaler = torch.cuda.amp.GradScaler() if args.use_amp else None\n',
        (
            'scaler = (\n'
            '        torch.cuda.amp.GradScaler()\n'
            "        if args.use_amp and configs['train_conf'].get('dtype') == 'fp16'\n"
            '        else None\n'
            '    )\n'
        ),
    )
    _replace_once(
        join_path,
        'autocast = torch.cuda.amp.autocast(enabled=scaler is not None, dtype=dtype)\n',
        (
            'autocast = torch.cuda.amp.autocast(\n'
            '            enabled=dtype in (torch.float16, torch.bfloat16),\n'
            '            dtype=dtype,\n'
            '        )\n'
        ),
    )


if __name__ == '__main__':
    main()
