#!/usr/bin/env python3
"""Attach non-lexical voice (cry, breath, sigh) to neighboring ASR clips."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import common


@dataclass
class Interval:
    """A time range in seconds on one source WAV."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load a mono float32 waveform."""
    audio, sample_rate = soundfile.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), int(sample_rate)


def energy_vad(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    min_voiced_sec: float = 0.15,
    merge_gap_sec: float = 0.30,
) -> list[Interval]:
    """Loose RMS VAD used only to recover non-lexical voice."""
    if audio.size == 0:
        return []
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, int(sample_rate * hop_ms / 1000.0))
    rms_values: list[float] = []
    for start in range(0, max(1, audio.size - frame_len + 1), hop_len):
        frame = audio[start:start + frame_len]
        rms_values.append(float(np.sqrt(np.mean(np.square(frame)) + 1e-12)))
    if not rms_values:
        return []
    rms = np.asarray(rms_values, dtype=np.float32)
    threshold = max(float(np.percentile(rms, 20)) * 1.8, float(np.median(rms)) * 0.35)
    voiced = rms >= threshold

    raw: list[Interval] = []
    in_seg = False
    seg_start = 0
    for index, flag in enumerate(voiced):
        if flag and not in_seg:
            in_seg = True
            seg_start = index
        elif not flag and in_seg:
            in_seg = False
            raw.append(Interval(seg_start * hop_ms / 1000.0, index * hop_ms / 1000.0))
    if in_seg:
        raw.append(Interval(seg_start * hop_ms / 1000.0, len(voiced) * hop_ms / 1000.0))

    merged: list[Interval] = []
    for interval in raw:
        if merged and interval.start - merged[-1].end <= merge_gap_sec:
            merged[-1] = Interval(merged[-1].start, interval.end)
        else:
            merged.append(interval)
    return [item for item in merged if item.duration >= min_voiced_sec]


def silero_vad(audio: np.ndarray, sample_rate: int) -> list[Interval] | None:
    """Optional Silero VAD; returns None if the package is unavailable."""
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError:
        return None
    model = load_silero_vad()
    # Silero expects 16 kHz; resample is skipped because extracts are 16 kHz.
    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sample_rate,
        threshold=0.3,
        min_speech_duration_ms=150,
        min_silence_duration_ms=200,
    )
    return [
        Interval(item['start'] / sample_rate, item['end'] / sample_rate)
        for item in timestamps
    ]


def subtract_covered(
    voiced: list[Interval],
    covered: list[Interval],
    pad_sec: float = 0.12,
) -> list[Interval]:
    """Keep voiced intervals that ASR did not already cover."""
    leftovers: list[Interval] = []
    for interval in voiced:
        cursor = interval.start
        for other in covered:
            left = other.start - pad_sec
            right = other.end + pad_sec
            if right <= cursor or left >= interval.end:
                continue
            if cursor < left:
                leftovers.append(Interval(cursor, min(left, interval.end)))
            cursor = max(cursor, right)
        if cursor < interval.end:
            leftovers.append(Interval(cursor, interval.end))
    return [item for item in leftovers if item.duration >= 0.12]


def choose_tag(interval: Interval, audio: np.ndarray, sample_rate: int) -> str:
    """Map a leftover voiced span to a CosyVoice3 paralinguistic tag."""
    start = max(0, int(interval.start * sample_rate))
    end = min(audio.size, int(interval.end * sample_rate))
    snippet = audio[start:end]
    rms = float(np.sqrt(np.mean(np.square(snippet)) + 1e-12)) if snippet.size else 0.0
    if interval.duration < 0.40 or rms < 0.02:
        return '[breath]'
    if interval.duration < 1.20:
        return '[sigh]'
    return '[vocalized-noise]'


def nearest_utterance(
    interval: Interval,
    utterances: list[dict[str, Any]],
) -> tuple[int, str] | None:
    """Find the closest utterance and which side the leftover sits on."""
    best_index = -1
    best_gap = 10**9
    best_side = 'after'
    for index, row in enumerate(utterances):
        start = float(row['start'])
        end = float(row['end'])
        if interval.end <= start:
            gap = start - interval.end
            side = 'before'
        elif interval.start >= end:
            gap = interval.start - end
            side = 'after'
        else:
            gap = 0.0
            side = 'before' if abs(interval.start - start) < abs(interval.end - end) else 'after'
        if gap < best_gap:
            best_gap = gap
            best_index = index
            best_side = side
    if best_index < 0:
        return None
    # Prefer merging into a neighbor. Only keep a standalone clip when the
    # leftover is long enough to be a real cry/sigh and far from any line.
    if best_gap > 3.0:
        return None
    return best_index, best_side


def apply_tag(text: str, tag: str, side: str) -> str:
    """Insert a CosyVoice tag without duplicating it."""
    if tag in text:
        return text
    if side == 'before':
        return f'{tag}{text}'
    return f'{text}{tag}'


def attach_for_source(
    source_wav: Path,
    utterances: list[dict[str, Any]],
    next_orphan_id: int,
) -> tuple[list[dict[str, Any]], int]:
    """Merge leftover voice into neighboring clips or keep tagged orphans."""
    audio, sample_rate = load_audio(source_wav)
    voiced = silero_vad(audio, sample_rate)
    if voiced is None:
        voiced = energy_vad(audio, sample_rate)
    covered = [Interval(float(row['start']), float(row['end'])) for row in utterances]
    leftovers = subtract_covered(voiced, covered)

    orphans: list[dict[str, Any]] = []
    for leftover in leftovers:
        tag = choose_tag(leftover, audio, sample_rate)
        match = nearest_utterance(leftover, utterances)
        if match is not None:
            index, side = match
            row = utterances[index]
            new_start = min(float(row['start']), leftover.start)
            new_end = max(float(row['end']), leftover.end)
            if new_end - new_start <= common.HARD_MAX_UTT_SEC:
                row['start'] = new_start
                row['end'] = new_end
            row['text'] = apply_tag(str(row['text']), tag, side)
            tags = list(row.get('para_tags') or [])
            if tag not in tags:
                tags.append(tag)
            row['para_tags'] = tags
            continue
        if leftover.duration < 0.8:
            continue

        next_orphan_id += 1
        source_index = int(utterances[0]['source_index']) if utterances else 0
        utt_id = f'vertin_{source_index:02d}_n{next_orphan_id:05d}'
        orphans.append({
            'utt_id': utt_id,
            'source_index': source_index,
            'source_name': utterances[0]['source_name'] if utterances else source_wav.name,
            'source_wav': common.relative_to_project(source_wav),
            'clip_wav': '',
            'start': round(leftover.start, 3),
            'end': round(leftover.end, 3),
            'duration': round(leftover.duration, 3),
            'text': tag,
            'para_tags': [tag],
            'sentences': [],
        })
    return orphans, next_orphan_id


def rewrite_clip(row: dict[str, Any]) -> None:
    """Re-export a clip after its time window changed."""
    source = common.PROJECT_ROOT / str(row['source_wav'])
    dest = common.CLIPS_DIR / f'{row["utt_id"]}.wav'
    duration = common.slice_wav(source, dest, float(row['start']), float(row['end']))
    row['clip_wav'] = common.relative_to_project(dest)
    row['duration'] = round(duration, 3)
    row['start'] = round(float(row['start']), 3)
    row['end'] = round(float(row['end']), 3)


def attach_nonspeech() -> list[dict[str, Any]]:
    """Update ASR clips with recovered non-lexical voice."""
    if not common.ASR_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f'{common.ASR_MANIFEST_PATH} is missing. Run 02_asr_cut.py first.'
        )
    rows = common.read_jsonl(common.ASR_MANIFEST_PATH)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row['source_wav']), []).append(row)

    updated: list[dict[str, Any]] = []
    next_orphan_id = 0
    for row in rows:
        sentences = row.get('sentences') or []
        if sentences:
            row['text'] = common.join_transcripts(
                item.get('text', '') for item in sentences
            )
    for source_rel, group in by_source.items():
        group.sort(key=lambda item: float(item['start']))
        source_path = common.PROJECT_ROOT / source_rel
        print(f'Attach nonspeech: {source_rel}')
        orphans, next_orphan_id = attach_for_source(source_path, group, next_orphan_id)
        combined = group + orphans
        for row in combined:
            rewrite_clip(row)
        updated.extend(combined)
        print(f'  {len(orphans)} orphan tagged clips, {len(combined)} total')

    updated.sort(key=lambda item: (int(item['source_index']), float(item['start'])))
    common.write_jsonl(common.ALL_MANIFEST_PATH, updated)
    print(f'Wrote {common.ALL_MANIFEST_PATH} ({len(updated)} clips)')
    return updated


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    attach_nonspeech()


if __name__ == '__main__':
    main()
