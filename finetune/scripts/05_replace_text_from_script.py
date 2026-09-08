#!/usr/bin/env python3
"""Replace ASR clip transcripts with Gnomon Vertin lines, keeping cut times."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import common

REPO = common.PROJECT_ROOT.parent
EXAMPLES = REPO / 'infer' / 'examples'
REPORT_PATH = common.MANIFEST_DIR / 'text_replace_report.tsv'

SOURCE_TO_SCRIPT = {
    2: EXAMPLES / '1st_in_our_time_vertin.txt',
    4: EXAMPLES / '2nd_tender_is_the_night_vertin.txt',
    3: EXAMPLES / '3rd_nouvelles_et_textes_pour_rien_vertin.txt',
    6: EXAMPLES / '4th_el_oro_de_los_tigres_vertin.txt',
    5: EXAMPLES / '5th_the_prisoner_in_the_cave_vertin.txt',
}

TAG_RE = re.compile(
    r'\[(?:breath|sigh|vocal|vocalized-noise|laughter|cry)\]',
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
YEAR_RE = re.compile(r'^(\d{4})s?$')

GLUE_PATTERNS = [
    (re.compile(r'\breg\s+ulus\b', re.I), 'Regulus'),
    (re.compile(r'\breg(?:ul)?\s+us\b', re.I), 'Regulus'),
    (re.compile(r'\bregul\s+us\b', re.I), 'Regulus'),
    (re.compile(r'\bregulars\b', re.I), 'Regulus'),
    (re.compile(r'\bson\s+ato\b', re.I), 'Sonetto'),
    (re.compile(r'\bson\s+ata\b', re.I), 'Sonetto'),
    (re.compile(r'\bson\s+net\s*to\b', re.I), 'Sonetto'),
    (re.compile(r'\bsen\s+eto\b', re.I), 'Sonetto'),
    (re.compile(r'\bsen\s+nato\b', re.I), 'Sonetto'),
    (re.compile(r'\bsennato\b', re.I), 'Sonetto'),
    (re.compile(r'\bsm\s+etto\b', re.I), 'Sonetto'),
    (re.compile(r'\bar\s+can\s+um\b', re.I), 'Arcanum'),
    (re.compile(r'\bar\s+can\s+ists?\b', re.I), 'arcanists'),
    (re.compile(r'\bf\s+ifty\b', re.I), 'Fifty'),
    (re.compile(r'\bpav\s+lov\b', re.I), 'Pavlov'),
    (re.compile(r'\bst\s*\.?\s*pavlov\b', re.I), 'St. Pavlov'),
    (re.compile(r'\btime\s+keeper\b', re.I), 'Timekeeper'),
    (re.compile(r'\bclass\s+mate\b', re.I), 'classmate'),
    (re.compile(r'\bsub\s+c\s+ulture\b', re.I), 'subculture'),
    (re.compile(r'\binjust\s+ices\b', re.I), 'injustices'),
    (re.compile(r'\bany\s+one\b', re.I), 'anyone'),
    (re.compile(r'\bcare\s+ful\b', re.I), 'careful'),
    (re.compile(r'\briot\s+ers\b', re.I), 'rioters'),
    (re.compile(r'\bass\s+ist\b', re.I), 'Assist'),
    (re.compile(r'\bcl\s+ose\b', re.I), 'Close'),
    (re.compile(r'\bf\s+ollow\b', re.I), 'Follow'),
    (re.compile(r'\ben\s+ough\b', re.I), 'Enough'),
    (re.compile(r'\bbra\s+ve\b', re.I), 'Brave'),
    (re.compile(r'\brel\s+ax\b', re.I), 'Relax'),
    (re.compile(r'\bal\s+so\b', re.I), 'Also'),
    (re.compile(r'\bex\s+actly\b', re.I), 'Exactly'),
    (re.compile(r'\bun\s+im\s+pro\s+ved\b', re.I), 'unimproved'),
    (re.compile(r'\bre\s+introdu\s+ce\b', re.I), 'reintroduce'),
    (re.compile(r'\bs\s+ame\b', re.I), 'Same'),
    (re.compile(r'\ber\s+as\b', re.I), 'eras'),
    (re.compile(r'\bclean\s+ers\b', re.I), 'cleaners'),
    (re.compile(r'\br\s+ave\s+llo\b', re.I), 'Ravello'),
    (re.compile(r'\bd\s+ating\b', re.I), 'dating'),
    (re.compile(r'\bmillenn\s+ium\b', re.I), 'millennium'),
    (re.compile(r'\bsuper\s+nova\b', re.I), 'supernova'),
    (re.compile(r'\beight\s+ies\b', re.I), 'eighties'),
    (re.compile(r'\bunve\s+iling\b', re.I), 'unveiling'),
    (re.compile(r'\bsixt\s+ies\b', re.I), 'sixties'),
    (re.compile(r'\bcrit\s+ters\b', re.I), 'critters'),
    (re.compile(r'\bn\s+eed\b', re.I), 'Need'),
    (re.compile(r'\ble\s+aving\b', re.I), 'Leaving'),
    (re.compile(r'\bse\s+ems\b', re.I), 'Seems'),
    (re.compile(r'\bsh\s+h\b', re.I), 'Shh'),
    (re.compile(r'\bmr\s+apple\b', re.I), 'Mr. Apple'),
    (re.compile(r'\ble\s+ave\b', re.I), 'leave'),
    (re.compile(r'\bh\s+ide\b', re.I), 'Hide'),
    (re.compile(r'\bm\s+r\b', re.I), 'Mr'),
    (re.compile(r'\bson\s+eto\b', re.I), 'Sonetto'),
    (re.compile(r'\bson\s+net\b', re.I), 'Sonetto'),
    (re.compile(r'\bman\s+us\b', re.I), 'Manus'),
    (re.compile(r'\bvind\s+ict(?:\s+ae|\s+ive)?\b', re.I), 'Vindictae'),
    (re.compile(r'\bschne\s+ider\b', re.I), 'Schneider'),
    (re.compile(r'\bsch\s+ne\s+ider\b', re.I), 'Schneider'),
    (re.compile(r'\bdru\s+vis\b', re.I), 'Druvis'),
    (re.compile(r'\bsothe\s+by\b', re.I), 'Sotheby'),
    (re.compile(r'\bward\s+en\b', re.I), 'Walden'),
    (re.compile(r'\bwoo\s+den\b', re.I), 'Walden'),
    (re.compile(r'\bdep\s+rive\b', re.I), 'deprive'),
    (re.compile(r'\bdif\s+ference\b', re.I), 'Difference'),
    (re.compile(r'\bcop\s+y\b', re.I), 'Copy'),
    (re.compile(r'\bmis\s+jud\s+gment\b', re.I), 'misjudgment'),
    (re.compile(r'\bhij\s+ack\b', re.I), 'hijack'),
    (re.compile(r'\bhost\s+ages\b', re.I), 'hostages'),
    (re.compile(r'\bl\s+abyrin\s+th\b', re.I), 'labyrinth'),
    (re.compile(r'\barc\s+ane\b', re.I), 'arcane'),
    (re.compile(r'\bfid\s+dle\b', re.I), 'fiddle'),
    (re.compile(r'\bper\s+usal\b', re.I), 'perusal'),
    (re.compile(r"\b(\w+)\s+'t\b", re.I), r"\1't"),
    (re.compile(r"\b(\w+)\s+n't\b", re.I), r"\1n't"),
    (re.compile(r"\b(\w+)\s+'s\b", re.I), r"\1's"),
    (re.compile(r"\b(\w+)\s+'m\b", re.I), r"\1'm"),
    (re.compile(r"\b(\w+)\s+'re\b", re.I), r"\1're"),
    (re.compile(r"\b(\w+)\s+'ve\b", re.I), r"\1've"),
    (re.compile(r"\b(\w+)\s+'ll\b", re.I), r"\1'll"),
    (re.compile(r"\b(\w+)\s+'d\b", re.I), r"\1'd"),
]

ONES = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
]
TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
DECADE_PLURAL = {
    20: 'twenties',
    30: 'thirties',
    40: 'forties',
    50: 'fifties',
    60: 'sixties',
    70: 'seventies',
    80: 'eighties',
    90: 'nineties',
}


@dataclass
class RenderTok:
    """A surface word in the official script."""

    raw: str
    start: int
    end: int


@dataclass
class MatchTok:
    """A normalized token used only for alignment."""

    norm: str
    render_index: int


def load_official_lines(path: Path) -> list[str]:
    """Read Vertin lines, skipping headers and stage comments."""
    lines = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        text = raw.strip()
        if not text or text.startswith('#') or text.startswith('Source:'):
            continue
        if '— Vertin lines only.' in text:
            continue
        if text.startswith('Gnomon ') or text.startswith('Stage directions'):
            continue
        lines.append(text)
    return lines


def two_digit_words(number: int) -> list[str]:
    """Convert 0-99 to English word tokens."""
    if number < 20:
        return [ONES[number]]
    tens, ones = divmod(number, 10)
    if ones == 0:
        return [TENS[tens]]
    return [TENS[tens], ONES[ones]]


def expand_year(raw: str) -> list[str] | None:
    """Expand 1999 / 1920s into spoken English tokens."""
    match = YEAR_RE.fullmatch(raw)
    if not match:
        return None
    year = int(match.group(1))
    century, rest = divmod(year, 100)
    head = two_digit_words(century)
    if raw.endswith('s'):
        decade = DECADE_PLURAL.get(rest)
        if decade:
            return head + [decade]
        return head + two_digit_words(rest)
    if rest == 0:
        return head + ['hundred']
    return head + two_digit_words(rest)


def glue_asr(text: str) -> str:
    """Merge common FunASR splits and spaced contractions."""
    cleaned = TAG_RE.sub(' ', text)
    cleaned = cleaned.replace("APPLe", "Apple")
    for pattern, repl in GLUE_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)
    return cleaned


def extract_tags(text: str) -> tuple[str, str, str]:
    """Split leading/trailing para tags from the spoken body."""
    prefix = ''
    suffix = ''
    body = text
    lead = re.match(r'^((?:' + TAG_RE.pattern + r'\s*)+)', body, re.I)
    if lead:
        prefix = lead.group(1)
        body = body[lead.end():]
    tail = re.search(r'((?:\s*' + TAG_RE.pattern + r')+)$', body, re.I)
    if tail:
        suffix = tail.group(1)
        body = body[:tail.start()]
    return prefix, body.strip(), suffix


def wrap_tags(prefix: str, body: str, suffix: str) -> str:
    """Put original para tags back around replacement text."""
    return f'{prefix}{body}{suffix}'


def official_streams(lines: list[str]) -> tuple[str, list[RenderTok], list[MatchTok]]:
    """Build render/match token streams from official lines."""
    blob = ' '.join(lines)
    renders: list[RenderTok] = []
    matches: list[MatchTok] = []
    for found in WORD_RE.finditer(blob):
        raw = found.group()
        render_index = len(renders)
        renders.append(RenderTok(raw=raw, start=found.start(), end=found.end()))
        expanded = expand_year(raw)
        if expanded:
            for word in expanded:
                matches.append(MatchTok(norm=word, render_index=render_index))
        else:
            matches.append(MatchTok(norm=raw.casefold(), render_index=render_index))
    return blob, renders, matches


def tokenize_asr(text: str) -> list[str]:
    """Normalize ASR words for matching."""
    glued = glue_asr(text)
    words = [found.group().casefold() for found in WORD_RE.finditer(glued)]
    return words


def is_smashed(words: list[str], raw_body: str = '') -> bool:
    """True when FunASR dumped a paragraph without spaces."""
    if raw_body and len(raw_body) >= 40 and raw_body.count(' ') < 4:
        return True
    if not words:
        return False
    if len(words) <= 2 and sum(len(word) for word in words) >= 40:
        return True
    return max(len(word) for word in words) >= 24


def bag_f1(left: list[str], right: list[str]) -> float:
    """Token-bag F1 between two word lists."""
    if not left or not right:
        return 0.0
    overlap = sum((Counter(left) & Counter(right)).values())
    precision = overlap / len(right)
    recall = overlap / len(left)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ordered_hits(asr_words: list[str], window: list[str]) -> float:
    """How many ASR words appear in order inside the official window."""
    index = 0
    hits = 0
    for word in asr_words:
        while index < len(window) and window[index] != word:
            index += 1
        if index < len(window) and window[index] == word:
            hits += 1
            index += 1
    return hits / max(len(asr_words), 1)


def score_window(asr_words: list[str], window: list[str]) -> float:
    """Fast hybrid score; SequenceMatcher is too slow for a full sweep."""
    if not asr_words or not window:
        return 0.0
    return 0.55 * bag_f1(asr_words, window) + 0.45 * ordered_hits(asr_words, window)


def best_word_window(
    asr_words: list[str],
    match_norms: list[str],
    cursor: int,
    search_ahead: int = 220,
) -> tuple[float, int, int]:
    """Find the official match-token span that best fits this clip."""
    count = len(asr_words)
    if count == 0 or not match_norms:
        return 0.0, cursor, cursor
    start_max = min(len(match_norms), max(cursor + 1, cursor + search_ahead))
    lengths = sorted({
        max(1, count - 8),
        max(1, count - 3),
        count,
        count + 4,
        count + 10,
        count + 16,
    })
    best = (0.0, cursor, cursor)
    for start in range(cursor, start_max):
        for length in lengths:
            end = min(len(match_norms), start + length)
            if end <= start:
                continue
            window = match_norms[start:end]
            if not anchors_ok(asr_words, window):
                continue
            score = score_window(asr_words, window)
            if asr_words and window and window[0] == asr_words[0]:
                score += 0.10
            if asr_words[:3] == window[:3]:
                score += 0.12
            elif window and window[0] not in asr_words[:6]:
                score -= 0.08
            score -= 0.00035 * (start - cursor)
            if score > best[0]:
                best = (score, start, end)
    return best


def best_char_window(
    asr_words: list[str],
    match_norms: list[str],
    cursor: int,
) -> tuple[float, int, int]:
    """Align smashed ASR by compact character matching."""
    asr_compact = re.sub(r'[^a-z0-9]', '', ''.join(asr_words))
    if len(asr_compact) < 8:
        return 0.0, cursor, cursor
    starts = [0]
    for word in match_norms:
        starts.append(starts[-1] + len(word))
    ref_compact = ''.join(match_norms)
    matcher = SequenceMatcher(None, asr_compact, ref_compact, autojunk=False)
    block = max(
        matcher.get_matching_blocks(),
        key=lambda item: item.size,
        default=None,
    )
    if block is None or block.size < 12:
        return 0.0, cursor, cursor

    def char_to_index(pos: int) -> int:
        for index, start in enumerate(starts[:-1]):
            if start <= pos < starts[index + 1]:
                return index
        return len(match_norms) - 1

    start = max(cursor, char_to_index(block.b))
    end = min(len(match_norms), char_to_index(block.b + block.size - 1) + 1)
    if end <= start:
        return 0.0, cursor, cursor
    score = block.size / max(len(asr_compact), 1)
    return score, start, end


def cap_end(
    start: int,
    end: int,
    duration: float,
    asr_count: int,
    words_per_sec: float = 3.3,
) -> int:
    """Keep official text close to what the clip could have said."""
    by_duration = max(6, int(duration * words_per_sec + 6))
    by_asr = max(asr_count + 4, int(asr_count * 1.2 + 3))
    max_words = min(by_duration, by_asr) if asr_count >= 8 else by_duration
    return min(end, start + max(6, max_words))


def snap_sentence_start(blob: str, renders: list[RenderTok], render_index: int) -> int:
    """If the clip starts mid-word, back up to the official sentence start."""
    start_char = renders[render_index].start
    prefix = blob[:start_char]
    last_break = max(prefix.rfind('. '), prefix.rfind('? '), prefix.rfind('! '))
    if last_break < 0:
        return render_index
    target = last_break + 2
    for index in range(render_index, -1, -1):
        if renders[index].start <= target:
            return index
    return render_index


def anchors_ok(asr_words: list[str], window: list[str]) -> bool:
    """Require a few longer ASR words to appear near the window start."""
    anchors = [word for word in asr_words[:10] if len(word) >= 4][:3]
    if len(anchors) < 2:
        return True
    head = set(window[:14])
    return sum(word in head for word in anchors) >= 2


def render_span(
    blob: str,
    renders: list[RenderTok],
    matches: list[MatchTok],
    start: int,
    end: int,
) -> str:
    """Recover official surface text for a matched token span."""
    if end <= start:
        return ''
    render_ids = [matches[index].render_index for index in range(start, end)]
    first = renders[render_ids[0]]
    last = renders[render_ids[-1]]
    return blob[first.start:last.end].strip()


def tidy_only(text: str) -> str:
    """Cleanup used for prologue and low-confidence matches."""
    prefix, body, suffix = extract_tags(text)
    body = glue_asr(body)
    body = re.sub(r'[\u3040-\u30ff\u4e00-\u9fff]+', ' ', body)
    body = re.sub(r'^[\s.]+', '', body)
    body = re.sub(r'\s+', ' ', body).strip()
    if body:
        body = body[0].upper() + body[1:]
    return wrap_tags(prefix, body, suffix)


def replace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return alignment rows for the ASR-cut speech clips."""
    by_source: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(int(row['source_index']), []).append(row)

    reports: list[dict[str, Any]] = []
    for source_index, group in by_source.items():
        group = sorted(group, key=lambda item: float(item['start']))
        script_path = SOURCE_TO_SCRIPT.get(source_index)
        if script_path is None:
            for row in group:
                new_text = tidy_only(str(row['text']))
                reports.append({
                    'utt_id': row['utt_id'],
                    'source_index': source_index,
                    'method': 'tidy',
                    'score': '',
                    'old': str(row['text']),
                    'new': new_text,
                })
                apply_text(row, new_text)
            continue

        lines = load_official_lines(script_path)
        blob, renders, matches = official_streams(lines)
        norms = [item.norm for item in matches]
        cursor = 0
        for row in group:
            prefix, body, suffix = extract_tags(str(row['text']))
            asr_words = tokenize_asr(body)
            duration = float(row.get('duration') or 0.0)
            if is_smashed(asr_words, body) and duration < 3.5:
                start = cursor
                end = min(len(norms), start + 40)
                score = 0.55
                method = 'char-prefix'
            elif is_smashed(asr_words, body):
                score, start, end = best_char_window(asr_words, norms, cursor)
                method = 'char'
            else:
                score, start, end = best_word_window(asr_words, norms, cursor)
                method = 'word'
            if start < end and asr_words:
                first_render = matches[start].render_index
                official_word = renders[first_render].raw.casefold()
                first_asr = asr_words[0]
                if (
                    official_word.endswith(first_asr)
                    and len(first_asr) >= 3
                    and len(first_asr) < len(official_word)
                ):
                    snapped = snap_sentence_start(blob, renders, first_render)
                    while start > 0 and matches[start - 1].render_index >= snapped:
                        start -= 1
            end = cap_end(start, end, duration, len(asr_words))
            official = render_span(blob, renders, matches, start, end)
            accept = bool(official) and score >= 0.44
            if accept:
                new_text = wrap_tags(prefix, official, suffix)
                cursor = max(cursor, end - 1)
            else:
                new_text = tidy_only(str(row['text']))
                method = f'{method}-fallback'
            reports.append({
                'utt_id': row['utt_id'],
                'source_index': source_index,
                'method': method,
                'score': f'{score:.3f}',
                'old': str(row['text']),
                'new': new_text,
            })
            apply_text(row, new_text)
    return reports


def apply_text(row: dict[str, Any], new_text: str) -> None:
    """Write replacement text and collapse word-level ASR sentences."""
    row['text'] = new_text
    spoken = TAG_RE.sub('', new_text).strip()
    row['sentences'] = [{
        'text': spoken or new_text,
        'start': row['start'],
        'end': row['end'],
    }]


def write_report(rows: list[dict[str, Any]]) -> None:
    """Write a review table of old vs new transcripts."""
    common.ensure_dir(REPORT_PATH.parent)
    lines = ['utt_id\tsource\tmethod\tscore\told\tnew']
    for row in rows:
        old = row['old'].replace('\t', ' ').replace('\n', ' ')
        new = row['new'].replace('\t', ' ').replace('\n', ' ')
        lines.append(
            f"{row['utt_id']}\t{row['source_index']}\t{row['method']}\t"
            f"{row['score']}\t{old}\t{new}"
        )
    REPORT_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def patch_all_manifest(replacements: dict[str, str]) -> int:
    """Update speech clips in all.jsonl; leave orphan tag clips alone."""
    if not common.ALL_MANIFEST_PATH.exists():
        return 0
    rows = common.read_jsonl(common.ALL_MANIFEST_PATH)
    updated = 0
    for row in rows:
        utt_id = str(row['utt_id'])
        if utt_id not in replacements:
            continue
        prefix, _, suffix = extract_tags(str(row.get('text', '')))
        body = TAG_RE.sub('', replacements[utt_id]).strip()
        apply_text(row, wrap_tags(prefix, body, suffix))
        if row.get('para_tags'):
            # Keep existing tags recorded by 03_attach_nonspeech.
            pass
        updated += 1
    common.write_jsonl(common.ALL_MANIFEST_PATH, rows)
    return updated


def patch_kaldi_text(replacements: dict[str, str]) -> int:
    """Rewrite Kaldi text files in place when present."""
    changed = 0
    for split in ('train', 'dev'):
        path = common.KALDI_DIR / split / 'text'
        if not path.exists():
            continue
        lines = []
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            utt_id, _, old = line.partition(' ')
            if utt_id in replacements:
                lines.append(f'{utt_id} {replacements[utt_id]}')
                changed += 1
            else:
                lines.append(line)
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print a sample and write the report without saving manifests.',
    )
    args = parser.parse_args()

    rows = common.read_jsonl(common.ASR_MANIFEST_PATH)
    reports = replace_rows(rows)
    write_report(reports)
    replacements = {item['utt_id']: item['new'] for item in reports}

    methods = {}
    for item in reports:
        methods[item['method']] = methods.get(item['method'], 0) + 1
    print(f'clips={len(reports)} methods={methods}')
    print(f'report={REPORT_PATH}')
    for item in reports[:6] + reports[-3:]:
        print(f"--- {item['utt_id']} {item['method']} {item['score']}")
        print(f"old: {item['old'][:160]}")
        print(f"new: {item['new'][:160]}")

    if args.dry_run:
        print('dry-run: manifests not written')
        return

    common.write_jsonl(common.ASR_MANIFEST_PATH, rows)
    all_n = patch_all_manifest(replacements)
    kaldi_n = patch_kaldi_text(replacements)
    print(f'wrote {common.ASR_MANIFEST_PATH}')
    print(f'updated all.jsonl={all_n} kaldi_text={kaldi_n}')
    print('parquet still has old text; re-run prepare_parquet.sh before next SFT')


if __name__ == '__main__':
    main()
