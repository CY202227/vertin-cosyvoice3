"""Shared helpers for Vertin CosyVoice3 data preparation."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000
SPEAKER_ID = 'vertin'
INSTRUCT_TEXT = 'You are a helpful assistant.<|endofprompt|>'
MIN_UTT_SEC = 5.0
MAX_UTT_SEC = 15.0
HARD_MAX_UTT_SEC = 20.0
MERGE_GAP_SEC = 1.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_WAV_DIR = DATA_DIR / 'raw_wav'
PROCESSED_DIR = DATA_DIR / 'processed'
CLIPS_DIR = PROCESSED_DIR / 'clips' / SPEAKER_ID
MANIFEST_DIR = PROCESSED_DIR / 'manifests'
KALDI_DIR = PROCESSED_DIR / 'kaldi'
SOURCE_MAP_PATH = RAW_WAV_DIR / 'source_map.jsonl'
ASR_MANIFEST_PATH = MANIFEST_DIR / 'asr_cuts.jsonl'
ALL_MANIFEST_PATH = MANIFEST_DIR / 'all.jsonl'


def ensure_dir(path: Path) -> Path:
    """Create a directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_source_videos(data_dir: Path = DATA_DIR) -> list[Path]:
    """Return every MP4 under data/, sorted by path for stable IDs."""
    videos = [
        path
        for path in data_dir.rglob('*')
        if path.is_file() and path.suffix.lower() == '.mp4'
    ]
    return sorted(videos, key=lambda path: path.as_posix().lower())


def sanitize_stem(name: str, fallback: str) -> str:
    """Make a filesystem-safe stem while keeping a readable hint."""
    cleaned = re.sub(r'[^\w\u4e00-\u9fff\-]+', '_', name, flags=re.UNICODE)
    cleaned = cleaned.strip('_')
    if not cleaned:
        cleaned = fallback
    return cleaned[:80]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write one JSON object per line using UTF-8."""
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write('\n')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file."""
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_ffmpeg() -> str:
    """Locate an ffmpeg binary, preferring PATH then imageio-ffmpeg."""
    from_path = shutil.which('ffmpeg')
    if from_path:
        return from_path
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            'ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.'
        ) from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg and raise if it fails."""
    command = [find_ffmpeg(), '-hide_banner', '-loglevel', 'error', *args]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f'ffmpeg failed: {detail}')


def extract_wav(src: Path, dest: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Extract 16 kHz mono PCM WAV from a video file."""
    ensure_dir(dest.parent)
    run_ffmpeg([
        '-y',
        '-i',
        str(src),
        '-vn',
        '-ac',
        '1',
        '-ar',
        str(sample_rate),
        '-c:a',
        'pcm_s16le',
        str(dest),
    ])


def slice_wav(
    src: Path,
    dest: Path,
    start_sec: float,
    end_sec: float,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """Cut [start, end] from src into dest. Returns duration in seconds."""
    if end_sec <= start_sec:
        raise ValueError(f'Invalid slice {start_sec=} {end_sec=} for {src}')
    duration = end_sec - start_sec
    ensure_dir(dest.parent)
    run_ffmpeg([
        '-y',
        '-ss',
        f'{start_sec:.3f}',
        '-i',
        str(src),
        '-t',
        f'{duration:.3f}',
        '-ac',
        '1',
        '-ar',
        str(sample_rate),
        '-c:a',
        'pcm_s16le',
        str(dest),
    ])
    return duration


def relative_to_project(path: Path) -> str:
    """Return a portable POSIX path relative to the repo root when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def join_transcripts(texts: Iterable[str]) -> str:
    """Join sentence texts; keep Chinese dense and English spaced."""
    parts = [item.strip() for item in texts if item and item.strip()]
    if not parts:
        return ''
    has_cjk = any(
        any('\u4e00' <= char <= '\u9fff' for char in part)
        for part in parts
    )
    return ''.join(parts) if has_cjk else ' '.join(parts)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL rows lazily."""
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
