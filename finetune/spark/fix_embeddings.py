#!/usr/bin/env python3
"""Cast CosyVoice embedding dicts to float32 before parquet export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def fix_file(path: Path) -> None:
    """Rewrite a .pt embedding map so values are numpy.float32 arrays."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f'Expected a dict in {path}, got {type(data)}')
    fixed = {}
    for key, value in data.items():
        if isinstance(value, list):
            fixed[key] = np.asarray(value, dtype=np.float32)
        elif isinstance(value, np.ndarray):
            fixed[key] = value.astype(np.float32, copy=False)
        elif torch.is_tensor(value):
            fixed[key] = value.detach().cpu().numpy().astype(np.float32)
        else:
            fixed[key] = np.asarray(value, dtype=np.float32)
    torch.save(fixed, path)
    print(f'Fixed {path} ({len(fixed)} keys)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.paths:
        fix_file(path)


if __name__ == '__main__':
    main()
