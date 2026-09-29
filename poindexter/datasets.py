"""SQuAD 2.0 and HotpotQA as gold-labeled records, plus the counterfactual swap.

Datasets are the original JSON files, downloaded once with urllib into `.poindexter/data/`
under the working directory, or into the directory named by $POINDEXTER_DATA.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import tempfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from poindexter.contract import Record, Unit, read_records, write_jsonl

DATA_ENV = "POINDEXTER_DATA"
UNANSWERABLE_SHARE = 0.3

_HOTPOT_OFFICIAL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

# name -> (urls tried in order, file name, sha256). The official HotpotQA host has been
# unreachable; the Wayback Machine raw snapshot serves the byte-identical file.
SOURCES: dict[str, tuple[tuple[str, ...], str, str]] = {
    "squad-dev": (
        ("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",),
        "dev-v2.0.json",
        "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    ),
    "squad-train": (
        ("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",),
        "train-v2.0.json",
        "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002",
    ),
    "hotpotqa-dev": (
        (_HOTPOT_OFFICIAL, "https://web.archive.org/web/20221010195146id_/" + _HOTPOT_OFFICIAL),
        "hotpot_dev_distractor_v1.json",
        "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea",
    ),
}


# --- download ----------------------------------------------------------------


def data_dir() -> Path:
    env = os.environ.get(DATA_ENV)
    return Path(env) if env else Path.cwd() / ".poindexter" / "data"


def fetch(name: str) -> Path:
    """Path to a source file, downloading and checksumming it on first use."""
    urls, filename, sha256 = SOURCES[name]
    target = data_dir() / filename
    if target.exists():
        got = _sha256(target)
        if got != sha256:
            raise ValueError(
                f"{target}: sha256 {got} != expected {sha256}; the cached file is corrupt or "
                "a different version. Delete it to re-download."
            )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        try:
            _download(url, target, sha256)
            return target
        except OSError as e:  # URLError, timeouts, dropped connections
            errors.append(f"{url}: {e}")
    raise RuntimeError(f"could not download {name}:\n" + "\n".join(errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path, sha256: str) -> None:
    # Write to a temp file in the same directory, then rename: several workspaces share it.
    digest = hashlib.sha256()
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
            while chunk := resp.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError(f"{url}: sha256 {digest.hexdigest()} != expected {sha256}")
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_squad(split: str = "dev", path: str | Path | None = None) -> dict[str, Any]:
    """The raw SQuAD 2.0 JSON for `split` (dev or train), or the file at `path`."""
    if split not in ("dev", "train"):
        raise ValueError(f"SQuAD split must be dev or train, got {split!r}")
    with open(path or fetch(f"squad-{split}")) as f:
        return json.load(f)


def load_hotpotqa(path: str | Path | None = None) -> list[dict[str, Any]]:
    """The raw HotpotQA dev distractor JSON, or the file at `path`."""
    with open(path or fetch("hotpotqa-dev")) as f:
        return json.load(f)


# --- sentence splitting ------------------------------------------------------

ABBREVIATIONS = frozenset(
    "mr mrs ms dr st no nos vs etc jr sr prof gen col lt sgt capt mt ft rev "
    "inc ltd co corp dept est approx fig vol ch pp op al c ca cf v "
    "jan feb mar apr jun jul aug sep sept oct nov dec".split()
)
# Terminal punctuation, closing quotes/brackets, Wikipedia footnote markers like [n 6], space.
_BOUNDARY = re.compile(r"[.!?]+[\"'”’)\]]*(?:\[[^\]\s]{0,3}(?: \d+)?\])*(\s+)")
_OPENERS = "\"'“‘(["
_DOTTED = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]")  # U.S, e.g, i.e, D.C


def sentence_split(text: str) -> list[tuple[int, int, str]]:
    """Split on [.!?] + whitespace + uppercase letter or digit, minus abbreviations.

    Returns (start, end, sentence) with text[start:end] == sentence, in order.
    """
    spans = []
    start = _skip_space(text, 0)
    for m in _BOUNDARY.finditer(text):
        nxt = m.end()
        while nxt < len(text) and text[nxt] in _OPENERS:
            nxt += 1
        if nxt >= len(text) or not (text[nxt].isupper() or text[nxt].isdigit()):
            continue
        if text[m.start()] == "." and _is_abbreviation(text, start, m.start()):
            continue
        end = m.start(1)
        spans.append((start, end, text[start:end]))
        start = m.end()
    end = len(text.rstrip())
    if start < end:
        spans.append((start, end, text[start:end]))
    return spans


def _skip_space(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def _is_abbreviation(text: str, start: int, dot: int) -> bool:
    words = text[start:dot].split()
    word = words[-1].lstrip(_OPENERS) if words else ""
    if word.lower() in ABBREVIATIONS or _DOTTED.fullmatch(word):
        return True
    return len(word) == 1 and word.isupper()  # an initial: "John F. Kennedy"


# --- SQuAD -------------------------------------------------------------------


def squad_records(data: dict[str, Any], split: str) -> list[Record]:
    """Every question in a SQuAD 2.0 JSON as a sentence-unit record, in dataset order."""
    records = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            spans = sentence_split(context)
            units = [Unit(f"u{i}", s) for i, (_, _, s) in enumerate(spans, 1)]
            for qa in para["qas"]:
                records.append(_squad_record(qa, context, spans, units, split, article["title"]))
    return records


def _squad_record(
    qa: dict[str, Any],
    context: str,
    spans: list[tuple[int, int, str]],
    units: list[Unit],
    split: str,
    title: str,
) -> Record:
    answers = qa["answers"]
    unanswerable = qa["is_impossible"]
    if unanswerable != (not answers):
        raise ValueError(f"squad {qa['id']}: is_impossible={unanswerable}, {len(answers)} answers")
    gold = set()
    for a in answers:
        lo, hi = a["answer_start"], a["answer_start"] + len(a["text"])
        if context[lo:hi] != a["text"]:
            raise ValueError(f"squad {qa['id']}: answer {a['text']!r} not at offset {lo}")
        gold |= {u.id for u, (s, e, _) in zip(units, spans, strict=True) if lo < e and hi > s}
    if answers and not gold:
        raise ValueError(f"squad {qa['id']}: answer spans map to no sentence")
    return Record(
        id=f"squad-{qa['id']}",
        question=qa["question"],
        units=units,
        gold=[u.id for u in units if u.id in gold],
        meta={
            "dataset": "squad",
            "split": split,
            "title": title,
            "unanswerable": unanswerable,
            "dataset_answers": list(dict.fromkeys(a["text"] for a in answers)),
            "answer_starts": [a["answer_start"] for a in answers],
            "sentence_spans": [[s, e] for s, e, _ in spans],
        },
    )


def sample_squad(records: list[Record], n: int, seed: int) -> list[Record]:
    """n records, 30% unanswerable (rounded), drawn and shuffled with random.Random(seed)."""
    unans = [r for r in records if r.meta["unanswerable"]]
    ans = [r for r in records if not r.meta["unanswerable"]]
    n_unans = round(n * UNANSWERABLE_SHARE)
    n_ans = n - n_unans
    if n_unans > len(unans) or n_ans > len(ans):
        raise ValueError(f"need {n_ans}+{n_unans}, have {len(ans)} answerable, {len(unans)} not")
    rng = random.Random(seed)
    out = rng.sample(ans, n_ans) + rng.sample(unans, n_unans)
    rng.shuffle(out)
    return out


# --- HotpotQA ----------------------------------------------------------------


def hotpot_records(data: list[dict[str, Any]], split: str = "dev") -> list[Record]:
    """Every HotpotQA question as a paragraph-unit record, in dataset order."""
    return [_hotpot_record(q, split) for q in data]


def _hotpot_record(q: dict[str, Any], split: str) -> Record:
    titles = [title for title, _ in q["context"]]
    if len(set(titles)) != len(titles):
        raise ValueError(f"hotpot {q['_id']}: duplicate paragraph titles")
    support = {title for title, _ in q["supporting_facts"]}
    if not support <= set(titles):
        raise ValueError(f"hotpot {q['_id']}: supporting titles {support - set(titles)} missing")
    units = [
        Unit(f"u{i}", f"{title}: {''.join(sents).strip()}")
        for i, (title, sents) in enumerate(q["context"], 1)
    ]
    return Record(
        id=f"hotpot-{q['_id']}",
        question=q["question"],
        units=units,
        gold=[u.id for u, t in zip(units, titles, strict=True) if t in support],
        meta={
            "dataset": "hotpotqa",
            "split": split,
            "type": q["type"],
            "level": q["level"],
            "dataset_answers": [q["answer"]],
        },
    )


def sample_hotpotqa(records: list[Record], n: int, seed: int) -> list[Record]:
    if n > len(records):
        raise ValueError(f"need {n} records, have {len(records)}")
    return random.Random(seed).sample(records, n)


# --- counterfactual: integer swap --------------------------------------------

_INTEGER = re.compile(r"[1-9]\d*")  # no leading zeros: "007" is a name
YEAR_CEILING = 2016  # SQuAD articles date from ~2016
YEAR_WINDOW = 25

FILTER_STEPS = (
    "all",
    "answerable",
    "first_answer_integer",
    "multi_digit",
    "answers_agree",
    "first_answer_in_one_sentence",
    "one_gold_sentence",
    "value_unique_in_units",
    "occurrence_is_answer_span",
    "swappable",
)

_NUMBER_WORDS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
# An integer mention: digits, optionally with thousands separators (1,000), not part of a
# decimal, a word (5th, A380), or a longer digit run; or a number word zero..twenty.
_MENTION = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?!\w|[.,]\d)"
    r"|\b(" + "|".join(_NUMBER_WORDS) + r")\b",
    re.IGNORECASE,
)


def number_mentions(text: str) -> list[tuple[int, int, int]]:
    """(start, end, value) for every integer mention: 1000, 1,000, three."""
    out = []
    for m in _MENTION.finditer(text):
        digits, word = m.groups()
        value = int(digits.replace(",", "")) if digits else _NUMBER_WORDS[word.lower()]
        out.append((m.start(), m.end(), value))
    return out


def _value_hits(value: int, units: list[Unit]) -> list[tuple[Unit, int, int]]:
    return [(u, s, e) for u in units for s, e, v in number_mentions(u.text) if v == value]


def _integer_step(r: Record) -> int:
    """How many FILTER_STEPS after `all` the record passes (9 means it is a candidate)."""
    if r.meta["unanswerable"]:
        return 0
    value = r.meta["dataset_answers"][0]
    if not _INTEGER.fullmatch(value):
        return 1
    if len(value) == 1:
        return 2
    v = int(value)
    if any([m[2] for m in number_mentions(a)] != [v] for a in r.meta["dataset_answers"]):
        return 3
    lo = r.meta["answer_starts"][0]
    spans = r.meta["sentence_spans"]
    inside = [i for i, (s, e) in enumerate(spans) if s <= lo and lo + len(value) <= e]
    if len(inside) != 1:
        return 4
    g = inside[0]
    if r.gold != [r.units[g].id]:
        return 5
    hits = _value_hits(v, r.units)
    if len(hits) != 1:
        return 6
    unit, start, end = hits[0]
    if unit.id != r.gold[0] or start != lo - spans[g][0] or unit.text[start:end] != value:
        return 7
    if not _swap_choices(value, "\n".join(u.text for u in r.units)):
        return 8
    return 9


def integer_filter_counts(records: Iterable[Record]) -> tuple[list[Record], dict[str, int]]:
    """Integer candidates among `records`, and how many records survive each filter step."""
    counts = dict.fromkeys(FILTER_STEPS, 0)
    out = []
    for r in records:
        passed = _integer_step(r)
        for step in FILTER_STEPS[: passed + 1]:
            counts[step] += 1
        if passed == len(FILTER_STEPS) - 1:
            out.append(r)
    return out, counts


def integer_candidates(split: str, path: str | Path | None = None) -> list[Record]:
    """Answerable SQuAD records whose integer answer can be swapped unambiguously.

    The first dataset answer is a bare integer of two or more digits without leading zeros;
    every annotated answer mentions exactly that one number; the first answer's span lies
    inside one sentence, which is the only gold unit; and the value is mentioned exactly once
    across all units in any form (1000, 1,000, three), as the digits at the annotated span;
    and swapped_integer has a plausible value for it. Dataset order.
    """
    return integer_filter_counts(squad_records(load_squad(split, path), split))[0]


def swapped_integer(original: str, rng: random.Random, avoid: str = "") -> str:
    """A plausible different integer with the same digit count, drawn from rng.

    Years (four digits in 1000..2100): original +/- 1..25, kept in 1000..max(2016, original)
    so no year moves past SQuAD's ~2016 articles. A side (earlier/later) is picked uniformly
    among the sides with a valid value, then a value uniformly on that side.
    Other n >= 10: uniform over [ceil(0.6n), floor(1.4n)] with the same digit count, over
    multiples of 10^z when n ends in z zeros (falling back to all integers in the window).
    One digit: uniform over 1..9.
    Never the original, and never a value `avoid` already mentions in any form (1,000, three).
    Raises when no value qualifies; integer_candidates excludes those records.
    """
    choices = _swap_choices(original, avoid)
    if not choices:
        raise ValueError(f"no plausible swap for {original}")
    if len(original) == 4 and 1000 <= int(original) <= 2100:
        return str(rng.choice(rng.choice(choices)))
    return str(rng.choice(choices[0]))


def _swap_choices(original: str, avoid: str) -> list[list[int]]:
    """The valid swap values, grouped: a year's earlier/later sides, else one group."""
    if not _INTEGER.fullmatch(original):
        raise ValueError(f"{original!r} is not an integer without leading zeros")
    v, d = int(original), len(original)
    present = {value for _, _, value in number_mentions(avoid)}

    def ok(x: int) -> bool:
        return len(str(x)) == d and x != v and x not in present

    if d == 4 and 1000 <= v <= 2100:
        top = max(YEAR_CEILING, v)
        sides = [
            [x for x in (v + sign * k for k in range(1, YEAR_WINDOW + 1)) if 1000 <= x <= top]
            for sign in (-1, 1)
        ]
        return [side for side in ([x for x in side if ok(x)] for side in sides) if side]
    if d == 1:
        pool = [x for x in range(1, 10) if ok(x)]
    else:
        lo, hi = -(-3 * v // 5), 7 * v // 5
        # Round numbers stay round: 8000 swaps to another multiple of 1000 when one fits.
        step = 10 ** (d - len(original.rstrip("0")))
        pool = [x for x in range(-(-lo // step) * step, hi + 1, step) if ok(x)]
        pool = pool or [x for x in range(lo, hi + 1) if ok(x)]
    return [pool] if pool else []


def swap_record(record: Record, seed: int) -> Record:
    """The record with its gold sentence's integer replaced. Deterministic per (seed, id).

    A supplied `answer` is dropped: it answered the unswapped text.
    """
    if record.gold is None or len(record.gold) != 1:
        raise ValueError(f"{record.id}: swap needs exactly one gold unit, got {record.gold}")
    original = record.meta["dataset_answers"][0]
    if not _INTEGER.fullmatch(original):
        raise ValueError(f"{record.id}: first answer {original!r} is not a bare integer")
    hits = _value_hits(int(original), record.units)
    if len(hits) != 1 or hits[0][0].id != record.gold[0]:
        raise ValueError(f"{record.id}: {original!r} is not mentioned once, in the gold unit only")
    unit, start, end = hits[0]
    if unit.text[start:end] != original:
        raise ValueError(f"{record.id}: gold unit writes {original!r} as {unit.text[start:end]!r}")
    # A swapped value already in the context would make the answer's source ambiguous.
    context = "\n".join(u.text for u in record.units)
    swapped = swapped_integer(original, random.Random(f"{seed}:{record.id}"), avoid=context)
    text = unit.text[:start] + swapped + unit.text[end:]
    meta = copy.deepcopy(record.meta)
    meta["counterfactual"] = {"original": original, "swapped": swapped, "source_id": record.id}
    meta["dataset_answers"] = [swapped]
    return Record(
        id=f"{record.id}~swap",
        question=record.question,
        units=[Unit(u.id, text) if u.id == unit.id else u for u in record.units],
        gold=list(record.gold),
        meta=meta,
    )


def redundant_record(record: Record, seed: int) -> Record:
    """The record with a copy of its gold unit inserted, never next to the original.

    The copy gets the next free id u<n>. Gold stays the original gold id list; the copy's
    id is in meta.redundant. Position is deterministic per (seed, id). A supplied `answer`
    is dropped: its cites refer to the context without the copy.
    """
    if record.gold is None or len(record.gold) != 1:
        raise ValueError(f"{record.id}: redundant copy needs one gold unit, got {record.gold}")
    ids = [u.id for u in record.units]
    g = ids.index(record.gold[0])
    # Inserting at index g or g + 1 would put the copy next to the original.
    allowed = [p for p in range(len(ids) + 1) if p not in (g, g + 1)]
    if not allowed:
        raise ValueError(f"{record.id}: too few units to place a non-adjacent copy")
    numbers = [int(i[1:]) for i in ids if re.fullmatch(r"u\d+", i)]
    copy_id = f"u{max(numbers, default=0) + 1}"
    units = list(record.units)
    units.insert(
        random.Random(f"{seed}:{record.id}:dup").choice(allowed), Unit(copy_id, units[g].text)
    )
    meta = copy.deepcopy(record.meta or {})
    meta["redundant"] = {"copy_of": record.gold[0], "copy_id": copy_id, "source_id": record.id}
    return Record(
        id=f"{record.id}~dup",
        question=record.question,
        units=units,
        gold=list(record.gold),
        meta=meta,
    )


# --- entry points ------------------------------------------------------------

KINDS = ("squad", "hotpotqa", "squad-int")


def build_records(
    kind: str,
    out: str | Path,
    *,
    split: str = "dev",
    n: int | None = None,
    seed: int = 0,
    source: str | Path | None = None,
) -> list[Record]:
    """Build records of `kind` and write them as JSONL to `out`.

    squad: stratified sample of n. hotpotqa: sample of n (dev only). squad-int: every
    integer candidate in dataset order (n must be None). source: a dataset JSON file to use
    instead of the download.
    """
    if kind == "squad":
        if n is None:
            raise ValueError("squad needs n")
        records = sample_squad(squad_records(load_squad(split, source), split), n, seed)
    elif kind == "hotpotqa":
        if split != "dev" or n is None:
            raise ValueError("hotpotqa needs split='dev' and n")
        records = sample_hotpotqa(hotpot_records(load_hotpotqa(source), split), n, seed)
    elif kind == "squad-int":
        if n is not None:
            raise ValueError("squad-int takes every candidate; n must be None")
        records = integer_candidates(split, source)
    else:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    write_jsonl(out, (r.to_json() for r in records))
    return records


def write_swapped(src: str | Path, out: str | Path, seed: int) -> list[Record]:
    """swap_record over every record in the JSONL at `src`, written to `out`."""
    records = [swap_record(r, seed) for r in read_records(src)]
    write_jsonl(out, (r.to_json() for r in records))
    return records


def write_redundant(src: str | Path, out: str | Path, seed: int) -> list[Record]:
    """redundant_record over the records in the JSONL at `src`, written to `out`.

    Single-unit records are skipped: there is nowhere to put a non-adjacent copy.
    """
    records = [redundant_record(r, seed) for r in read_records(src) if len(r.units) > 1]
    write_jsonl(out, (r.to_json() for r in records))
    return records
