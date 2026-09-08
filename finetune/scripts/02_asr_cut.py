#!/usr/bin/env python3
"""Cut long WAV files into sentence clips with timestamped ASR."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import common


@dataclass
class Sentence:
    """One ASR sentence with times in seconds."""

    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _wav_duration(path: Path) -> float:
    """Return WAV duration in seconds."""
    import soundfile

    return float(soundfile.info(str(path)).duration)


def _pairs_to_seconds(
    pairs: list[tuple[float, float]],
    audio_duration: float,
) -> list[tuple[float, float]]:
    """Normalize FunASR [start, end] pairs to seconds.

    Fun-ASR-Nano writes integer milliseconds. Values such as 840 must not be
    treated as 840 seconds just because they are <= 1000.
    """
    if not pairs:
        return []
    max_value = max(max(start, end) for start, end in pairs)
    as_ms = max_value > (audio_duration + 1.0) * 2
    if not as_ms and max_value > 1000 and max_value / 1000.0 <= audio_duration + 2.0:
        as_ms = True
    scale = 0.001 if as_ms else 1.0
    converted: list[tuple[float, float]] = []
    for start, end in pairs:
        start_s = float(start) * scale
        end_s = float(end) * scale
        start_s = max(0.0, min(start_s, audio_duration))
        end_s = max(0.0, min(end_s, audio_duration))
        if end_s > start_s:
            converted.append((start_s, end_s))
    return converted


def _ms_to_sec(value: float, audio_duration: float | None = None) -> float:
    """Convert one FunASR timestamp to seconds."""
    value = float(value)
    if audio_duration and audio_duration > 0:
        if value > audio_duration + 1.0 and value / 1000.0 <= audio_duration + 1.0:
            return value / 1000.0
    elif value > 1000:
        return value / 1000.0
    return value


def load_funasr_model():
    """Load FunASR paraformer-zh with punctuation and VAD."""
    from funasr import AutoModel

    return AutoModel(
        model='paraformer-zh',
        vad_model='fsmn-vad',
        punc_model='ct-punc',
        disable_update=True,
    )


def load_funasr_nano(device: str = 'cuda:0'):
    """Load Fun-ASR-Nano-2512 with VAD for long English files."""
    from funasr import AutoModel

    return AutoModel(
        model='FunAudioLLM/Fun-ASR-Nano-2512',
        trust_remote_code=True,
        vad_model='fsmn-vad',
        vad_kwargs={'max_single_segment_time': 15000},
        device=device,
        hub='ms',
        disable_update=True,
    )


def _nano_language(language: str) -> str | None:
    """Map CLI language codes to Fun-ASR-Nano language names."""
    if language == 'en':
        return '英文'
    if language == 'zh':
        return '中文'
    return None


def _sentences_from_word_timestamps(
    items: list[Any],
    audio_duration: float,
    fallback_words: list[str] | None = None,
) -> list[Sentence]:
    """Turn per-token timestamps into sentences for 5-15s packing."""
    raw_pairs: list[tuple[float, float]] = []
    texts: list[str] = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            text = str(item.get('token') or item.get('text') or '').strip()
            start = item.get('start_time', item.get('start'))
            end = item.get('end_time', item.get('end'))
            if start is None or end is None:
                continue
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
            if fallback_words and index < len(fallback_words):
                text = fallback_words[index]
            else:
                text = ''
        else:
            continue
        if text in {'<|nospeech|>', '/sil', '<sil>'}:
            continue
        raw_pairs.append((float(start), float(end)))
        texts.append(text)
    if not raw_pairs:
        return []
    max_value = max(max(start, end) for start, end in raw_pairs)
    as_ms = max_value > (audio_duration + 1.0) * 2
    if not as_ms and max_value > 1000 and max_value / 1000.0 <= audio_duration + 2.0:
        as_ms = True
    scale = 0.001 if as_ms else 1.0
    sentences: list[Sentence] = []
    for (start, end), text in zip(raw_pairs, texts):
        start_s = max(0.0, min(float(start) * scale, audio_duration))
        end_s = max(0.0, min(float(end) * scale, audio_duration))
        if end_s <= start_s:
            continue
        sentences.append(Sentence(text=text or '[vocal]', start=start_s, end=end_s))
    return sentences


def _sentences_from_funasr_payload(
    payload: dict[str, Any],
    audio_duration: float,
) -> list[Sentence]:
    """Parse FunASR / Fun-ASR-Nano generate() output into timed sentences."""
    nano_ts = payload.get('timestamps')
    if isinstance(nano_ts, list) and nano_ts and isinstance(nano_ts[0], dict):
        sentences = _sentences_from_word_timestamps(nano_ts, audio_duration)
        if sentences:
            return sentences

    sentences: list[Sentence] = []
    for item in payload.get('sentence_info') or []:
        text = str(item.get('text') or item.get('sentence') or '').strip()
        if not text:
            continue
        start = item.get('start', item.get('start_time', 0.0))
        end = item.get('end', item.get('end_time', 0.0))
        pairs = _pairs_to_seconds([(float(start), float(end))], audio_duration)
        if pairs:
            sentences.append(Sentence(text=text, start=pairs[0][0], end=pairs[0][1]))
    if sentences:
        return sentences

    text = str(payload.get('text', '')).strip()
    words = payload.get('words')
    word_list = (
        [str(word).strip() for word in words if str(word).strip()]
        if isinstance(words, list)
        else text.split()
    )
    timestamp = payload.get('timestamp') or []
    if timestamp:
        sentences = _sentences_from_word_timestamps(
            timestamp, audio_duration, fallback_words=word_list
        )
        if sentences:
            if not any(item.text and item.text != '[vocal]' for item in sentences):
                if len(word_list) == len(sentences):
                    for sentence, word in zip(sentences, word_list):
                        sentence.text = word
                elif text:
                    sentences[0].text = text
                    for sentence in sentences[1:]:
                        sentence.text = ''
            return sentences
    if text:
        return [Sentence(text=text, start=0.0, end=audio_duration)]
    return []


def transcribe_funasr_nano(
    model,
    wav_path: Path,
    language: str = 'en',
) -> list[Sentence]:
    """Run Fun-ASR-Nano. VAD segment times are used for cutting."""
    kwargs: dict[str, Any] = {
        'input': [str(wav_path)],
        'cache': {},
        'batch_size': 1,
        'itn': True,
    }
    nano_lang = _nano_language(language)
    if nano_lang:
        kwargs['language'] = nano_lang
    results = model.generate(**kwargs)
    if not results:
        return []
    audio_duration = _wav_duration(wav_path)
    sentences = _sentences_from_funasr_payload(results[0], audio_duration)
    print(
        f'  funasr-nano language={nano_lang or "auto"} '
        f'duration={audio_duration:.1f}s stamps={len(sentences)}',
        flush=True,
    )
    return sentences


def transcribe_funasr(model, wav_path: Path) -> list[Sentence]:
    """Run FunASR with sentence timestamps."""
    results = model.generate(
        input=str(wav_path),
        batch_size_s=300,
        sentence_timestamp=True,
    )
    if not results:
        return []
    payload = results[0]
    sentences: list[Sentence] = []
    for item in payload.get('sentence_info') or []:
        text = str(item.get('text', '')).strip()
        start = _ms_to_sec(float(item['start']), audio_duration=None)
        end = _ms_to_sec(float(item['end']), audio_duration=None)
        if text and end > start:
            sentences.append(Sentence(text=text, start=start, end=end))
    if sentences:
        return sentences

    text = str(payload.get('text', '')).strip()
    timestamp = payload.get('timestamp') or []
    if text and timestamp:
        start = _ms_to_sec(float(timestamp[0][0]), audio_duration=None)
        end = _ms_to_sec(float(timestamp[-1][1]), audio_duration=None)
        return [Sentence(text=text, start=start, end=end)]
    return []


def load_whisper_model(model_size: str):
    """Load faster-whisper once for all source files."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device='cpu', compute_type='int8')


def drop_repeat_hallucinations(sentences: list[Sentence]) -> list[Sentence]:
    """Drop Whisper loops that repeat the same phrase many times."""
    cleaned: list[Sentence] = []
    last_text = ''
    repeat = 0
    for sentence in sentences:
        key = sentence.text.strip().casefold()
        if key == last_text:
            repeat += 1
            if repeat > 2:
                continue
        else:
            last_text = key
            repeat = 1
        cleaned.append(sentence)
    return cleaned


def transcribe_whisper(
    model,
    wav_path: Path,
    language: str = 'en',
) -> list[Sentence]:
    """ASR using faster-whisper segment timestamps."""
    kwargs = {
        'word_timestamps': False,
        'vad_filter': True,
        'beam_size': 5,
        'condition_on_previous_text': False,
        'without_timestamps': False,
    }
    if language != 'auto':
        kwargs['language'] = language
    segments, info = model.transcribe(str(wav_path), **kwargs)
    detected = getattr(info, 'language', language)
    print(f'  whisper language={detected}', flush=True)
    sentences: list[Sentence] = []
    for segment in segments:
        text = (segment.text or '').strip()
        if text and segment.end > segment.start:
            sentences.append(
                Sentence(text=text, start=float(segment.start), end=float(segment.end))
            )
    return drop_repeat_hallucinations(sentences)


def pack_sentences(
    sentences: list[Sentence],
    min_dur: float = common.MIN_UTT_SEC,
    max_dur: float = common.MAX_UTT_SEC,
    hard_max: float = common.HARD_MAX_UTT_SEC,
) -> list[list[Sentence]]:
    """Pack sentences into roughly 5-15s groups without dropping text."""
    groups: list[list[Sentence]] = []
    current: list[Sentence] = []

    def current_span() -> float:
        return current[-1].end - current[0].start

    for sentence in sentences:
        if sentence.duration > hard_max:
            if current:
                groups.append(current)
                current = []
            groups.append([sentence])
            continue
        if not current:
            current = [sentence]
            continue
        new_span = sentence.end - current[0].start
        cur_span = current_span()
        if new_span <= max_dur:
            current.append(sentence)
            continue
        if cur_span < min_dur and new_span <= hard_max:
            current.append(sentence)
            continue
        groups.append(current)
        current = [sentence]
    if current:
        groups.append(current)
    return groups


def split_overlong_group(
    group: list[Sentence],
    hard_max: float = common.HARD_MAX_UTT_SEC,
) -> list[list[Sentence]]:
    """Split a packed group that still exceeds the hard max."""
    span = group[-1].end - group[0].start
    if span <= hard_max or len(group) == 1:
        return [group]
    mid = max(1, len(group) // 2)
    left = split_overlong_group(group[:mid], hard_max)
    right = split_overlong_group(group[mid:], hard_max)
    return left + right


def cut_one_wav(
    source_row: dict[str, Any],
    sentences: list[Sentence],
    clips_dir: Path,
    utt_start: int,
) -> list[dict[str, Any]]:
    """Write clips and manifest rows for one source WAV."""
    raw_wav = common.PROJECT_ROOT / str(source_row['raw_wav'])
    source_index = int(source_row['source_index'])
    groups: list[list[Sentence]] = []
    for group in pack_sentences(sentences):
        groups.extend(split_overlong_group(group))

    rows: list[dict[str, Any]] = []
    for offset, group in enumerate(groups, start=1):
        utt_id = f'vertin_{source_index:02d}_{utt_start + offset:05d}'
        start = float(group[0].start)
        end = float(group[-1].end)
        if end <= start:
            print(f'  skip invalid slice {utt_id} {start:.3f}->{end:.3f}', flush=True)
            continue
        text = common.join_transcripts(item.text for item in group)
        clip_path = clips_dir / f'{utt_id}.wav'
        duration = common.slice_wav(raw_wav, clip_path, start, end)
        rows.append({
            'utt_id': utt_id,
            'source_index': source_index,
            'source_name': source_row['source_name'],
            'source_wav': source_row['raw_wav'],
            'clip_wav': common.relative_to_project(clip_path),
            'start': round(start, 3),
            'end': round(end, 3),
            'duration': round(duration, 3),
            'text': text,
            'para_tags': [],
            'sentences': [
                {'text': item.text, 'start': item.start, 'end': item.end}
                for item in group
            ],
        })
    return rows


def run_asr_cut(
    backend: str,
    whisper_model: str,
    language: str = 'en',
    reset_clips: bool = True,
) -> list[dict[str, Any]]:
    """ASR-cut every extracted source WAV."""
    common.find_ffmpeg()
    if not common.SOURCE_MAP_PATH.exists():
        raise FileNotFoundError(
            f'{common.SOURCE_MAP_PATH} is missing. Run 01_extract_audio.py first.'
        )
    sources = common.read_jsonl(common.SOURCE_MAP_PATH)
    common.ensure_dir(common.CLIPS_DIR)
    common.ensure_dir(common.MANIFEST_DIR)
    if reset_clips:
        for old_clip in common.CLIPS_DIR.glob('*.wav'):
            old_clip.unlink()

    funasr_model = None
    whisper_handle = None
    if backend == 'funasr-nano':
        device = 'cuda:0'
        try:
            import torch
            if not torch.cuda.is_available():
                device = 'cpu'
        except Exception:
            device = 'cpu'
        print(f'Loading Fun-ASR-Nano-2512 on {device} ...', flush=True)
        funasr_model = load_funasr_nano(device=device)
    elif backend == 'funasr':
        try:
            print('Loading FunASR paraformer-zh ...')
            funasr_model = load_funasr_model()
        except Exception as error:
            print(f'FunASR unavailable ({error}). Falling back to faster-whisper.')
            backend = 'whisper'
    if backend == 'whisper':
        print(f'Loading faster-whisper {whisper_model} ...')
        whisper_handle = load_whisper_model(whisper_model)

    all_rows: list[dict[str, Any]] = []
    next_utt = 0
    for source in sources:
        wav_rel = str(source['raw_wav'])
        wav_path = common.PROJECT_ROOT / wav_rel
        print(f'ASR {source["source_name"]} ({wav_rel})', flush=True)
        if backend == 'funasr-nano':
            sentences = transcribe_funasr_nano(
                funasr_model, wav_path, language=language
            )
        elif backend == 'funasr':
            sentences = transcribe_funasr(funasr_model, wav_path)
        else:
            sentences = transcribe_whisper(
                whisper_handle, wav_path, language=language
            )
        print(f'  {len(sentences)} sentences', flush=True)
        rows = cut_one_wav(source, sentences, common.CLIPS_DIR, next_utt)
        next_utt += len(rows)
        all_rows.extend(rows)
        common.write_jsonl(common.ASR_MANIFEST_PATH, all_rows)
        print(f'  {len(rows)} clips', flush=True)

    print(f'Wrote {common.ASR_MANIFEST_PATH} ({len(all_rows)} clips)', flush=True)
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--backend',
        choices=('funasr-nano', 'funasr', 'whisper'),
        default='funasr-nano',
        help='ASR backend used for timestamps and transcripts.',
    )
    parser.add_argument(
        '--whisper-model',
        default='distil-large-v3',
        help='English-only faster-whisper model (distil-large-v3 / medium.en).',
    )
    parser.add_argument(
        '--language',
        default='en',
        choices=('en', 'zh', 'auto'),
        help='Decode language. These Vertin cuts are English audio.',
    )
    parser.add_argument(
        '--keep-clips',
        action='store_true',
        help='Do not delete existing clips before recutting.',
    )
    args = parser.parse_args()
    run_asr_cut(
        args.backend,
        args.whisper_model,
        language=args.language,
        reset_clips=not args.keep_clips,
    )


if __name__ == '__main__':
    main()
