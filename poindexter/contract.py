"""Records, model answers, the validator, and the result schema every module shares."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MAX_ANSWER_CHARS = 200

SAME = "same"
ABSTAIN = "abstain"
OTHER = "other"
UNSTABLE = "unstable"
# Probe C only: the sample's answer is the counterfactual replacement value.
FOLLOWS = "follows"

# Probe keys in Result.probes. Leave-one-out runs live in Result.loo, keyed by unit id.
PROBES = ("O", "N", "R_remove", "R_replace", "M", "S")

REJECTION_CODES = ("INVALID_JSON", "BAD_KEYS", "BAD_ABSTAIN", "BAD_ANSWER", "BAD_CITES")


@dataclass(frozen=True)
class Unit:
    id: str
    text: str


@dataclass(frozen=True)
class Answer:
    text: str | None
    cites: list[str]
    abstain: bool

    def to_json(self) -> dict[str, Any]:
        return {"text": self.text, "cites": list(self.cites), "abstain": self.abstain}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Answer:
        _require_keys(d, {"text", "cites", "abstain"}, set(), "answer")
        text, cites, abstain = d["text"], d["cites"], d["abstain"]
        if text is not None and not isinstance(text, str):
            raise ValueError(f"answer: text must be a string or null, got {text!r}")
        if not isinstance(cites, list) or not all(isinstance(c, str) for c in cites):
            raise ValueError(f"answer: cites must be a list of strings, got {cites!r}")
        if not isinstance(abstain, bool):
            raise ValueError(f"answer: abstain must be a bool, got {abstain!r}")
        return cls(text=text, cites=list(cites), abstain=abstain)


@dataclass(frozen=True)
class Record:
    id: str
    question: str
    units: list[Unit]
    answer: Answer | None = None
    gold: list[str] | None = None
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        ids = [u.id for u in self.units]
        if len(ids) != len(set(ids)):
            raise ValueError(f"record {self.id}: unit id collision in {ids}")
        if self.answer is not None:
            # cites [] is allowed here; only the open agent may be audited with it, which
            # run_record enforces, since a Record does not know its agent.
            checked = _check_answer(
                self.answer.text, self.answer.cites, self.answer.abstain, self.unit_ids,
                uncited_ok=True,
            )
            if isinstance(checked, Rejection):
                raise ValueError(
                    f"record {self.id}: supplied answer violates the contract: "
                    f"{checked.code}: {checked.message}"
                )

    @property
    def unit_ids(self) -> set[str]:
        return {u.id for u in self.units}

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "question": self.question,
            "units": [asdict(u) for u in self.units],
        }
        if self.answer is not None:
            d["answer"] = self.answer.to_json()
        if self.gold is not None:
            d["gold"] = list(self.gold)
        if self.meta is not None:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Record:
        _require_keys(d, {"id", "question", "units"}, {"answer", "gold", "meta"}, "record")
        units = []
        for u in d["units"]:
            _require_keys(u, {"id", "text"}, set(), "unit")
            units.append(Unit(id=u["id"], text=u["text"]))
        if "answer" in d and d["answer"] is None:
            raise ValueError(f"record {d['id']}: answer is null; omit the key to generate one")
        answer = Answer.from_json(d["answer"]) if "answer" in d else None
        return cls(
            id=d["id"],
            question=d["question"],
            units=units,
            answer=answer,
            gold=d.get("gold"),
            meta=d.get("meta"),
        )


def _require_keys(d: dict[str, Any], required: set[str], optional: set[str], what: str) -> None:
    keys = set(d)
    missing = required - keys
    unknown = keys - required - optional
    if missing or unknown:
        raise ValueError(f"{what}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_records(path: str | Path) -> list[Record]:
    return [Record.from_json(d) for d in read_jsonl(path)]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# --- validator ---------------------------------------------------------------


@dataclass(frozen=True)
class Rejection:
    code: str
    message: str


_FENCE = re.compile(r"\A```json[ \t]*\n(.*)\n```\Z", re.DOTALL)


def _extract_object(raw: str) -> dict[str, Any] | Rejection:
    s = raw.strip()
    m = _FENCE.match(s)
    body = m.group(1) if m else s
    duplicates: list[str] = []

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [k for k, _ in items]
        duplicates.extend(k for k in set(keys) if keys.count(k) > 1)
        return dict(items)

    try:
        obj = json.loads(body, object_pairs_hook=pairs)
    except json.JSONDecodeError as e:
        return Rejection("INVALID_JSON", f"not one JSON object ({e.msg})")
    if not isinstance(obj, dict):
        return Rejection("INVALID_JSON", "top-level value is not an object")
    if duplicates:
        return Rejection("BAD_KEYS", f"duplicate keys {sorted(set(duplicates))}")
    return obj


def validate(raw: str, unit_ids: set[str]) -> Answer | Rejection:
    return _validate(raw, unit_ids, uncited_ok=False)


def validate_open(raw: str, unit_ids: set[str]) -> Answer | Rejection:
    """The open agent's validator: `validate`, except that an answer may cite nothing.

    The open agent may answer from its own knowledge, and `cites: []` is how it says so.
    Every other rule is unchanged.
    """
    return _validate(raw, unit_ids, uncited_ok=True)


def _validate(raw: str, unit_ids: set[str], uncited_ok: bool) -> Answer | Rejection:
    obj = _extract_object(raw)
    if isinstance(obj, Rejection):
        return obj
    if set(obj) != {"answer", "cites", "abstain"}:
        got = sorted(obj)
        return Rejection("BAD_KEYS", f"keys must be exactly answer, cites, abstain; got {got}")
    return _check_answer(obj["answer"], obj["cites"], obj["abstain"], unit_ids, uncited_ok)


def _check_answer(
    answer: Any, cites: Any, abstain: Any, unit_ids: set[str], uncited_ok: bool = False
) -> Answer | Rejection:
    """The contract rules on already-extracted values, shared by validate and supplied answers.

    uncited_ok lets a non-abstaining answer have empty cites (the open agent only)."""
    if not isinstance(abstain, bool):
        return Rejection("BAD_ABSTAIN", "abstain must be true or false")
    if abstain:
        if answer is not None or cites != []:
            msg = "abstain is true, so answer must be null and cites empty"
            return Rejection("BAD_ABSTAIN", msg)
        return Answer(text=None, cites=[], abstain=True)
    if not isinstance(cites, list) or not all(isinstance(c, str) for c in cites):
        return Rejection("BAD_CITES", "cites must be a list of unit id strings")
    if answer is not None and not isinstance(answer, str):
        return Rejection("BAD_ANSWER", "answer must be a string or null")
    if answer is None or not answer.strip():
        return Rejection("BAD_ANSWER", "abstain is false, so answer must be a non-empty string")
    if len(answer) > MAX_ANSWER_CHARS:
        return Rejection("BAD_ANSWER", f"answer is over {MAX_ANSWER_CHARS} characters")
    if not cites and not uncited_ok:
        return Rejection("BAD_CITES", "abstain is false, so cites must not be empty")
    if len(cites) != len(set(cites)):
        return Rejection("BAD_CITES", "cites has duplicates")
    unknown = [c for c in cites if c not in unit_ids]
    if unknown:
        return Rejection("BAD_CITES", f"cites has ids not in the context: {unknown}")
    return Answer(text=answer, cites=cites, abstain=False)


# --- equivalence -------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    """The SQuAD v2 official answer normalizer."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()
_THOUSANDS = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_NUMBER = re.compile(
    r"\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE
)
# A magnitude word changes the value ("one hundred", "1 million"), so such text is not
# reduced to its single number.
_MAGNITUDE = re.compile(
    r"\b(?:hundred|thousand|million|billion|trillion|dozen|score)s?\b", re.IGNORECASE
)
_MINUS = "-\u2212"


def canonical(s: str) -> str:
    """The equivalence key for answers.

    Text holding exactly one number (a digit run, thousands separators folded, a decimal
    kept as written, or a number word zero..twenty) is "#<number>", so "two atoms",
    "in 2", and "2" are equal. Anything else is normalize(s), including text with a
    magnitude word ("one hundred", "1 million") or a minus sign attached to the number
    ("-1"). The "#" cannot occur in normalize output, so the two kinds never collide.
    """
    folded = _THOUSANDS.sub(lambda m: m.group().replace(",", ""), s)
    numbers = list(_NUMBER.finditer(folded))
    if len(numbers) != 1 or _MAGNITUDE.search(folded):
        return normalize(s)
    start = numbers[0].start()
    if start > 0 and folded[start - 1] in _MINUS:
        return normalize(s)
    n = numbers[0].group().lower()
    if n in _NUMBER_WORDS:
        return f"#{_NUMBER_WORDS.index(n)}"
    return f"#{n}" if "." in n else f"#{int(n)}"


VALUE_SEPARATOR = " / "


def split_values(text: str) -> list[str]:
    """The canonical() of each value in an answer that lists several, separated by " / "
    as the prompts ask for conflicting context: "1527 / 1525" -> ["#1527", "#1525"].
    A single-valued answer gives one element; empty parts are dropped."""
    parts = [p for p in text.split(VALUE_SEPARATOR) if p.strip()]
    return [canonical(p) for p in parts] or [canonical(text)]


def span_in_cites(answer: Answer, units: list[Unit]) -> bool:
    if answer.text is None or not answer.cites:
        return False
    cited = " ".join(u.text for u in units if u.id in set(answer.cites))
    return normalize(answer.text) in normalize(cited)


def outcome_of(sample: Answer | None, a: Answer) -> str:
    """Outcome of one probe sample relative to the original answer A.

    None is a sample rejected twice by the validator; it counts as `other`.
    Equivalence is `canonical` equality.
    """
    if sample is None:
        return OTHER
    if sample.abstain:
        return ABSTAIN
    if a.text is not None and canonical(sample.text or "") == canonical(a.text):
        return SAME
    return OTHER


def counterfactual_outcome(sample: Answer | None, a: Answer, replacement: str) -> str:
    """Outcome of one probe C sample: `follows` if it answers the replacement value,
    `same` if it answers A, else `abstain` or `other` (rejected twice is `other`)."""
    if sample is None:
        return OTHER
    if sample.abstain:
        return ABSTAIN
    got = canonical(sample.text or "")
    if got == canonical(replacement):
        return FOLLOWS
    if a.text is not None and got == canonical(a.text):
        return SAME
    return OTHER


def majority(outcomes: list[str]) -> str:
    """Strict majority over samples, else `unstable`."""
    if not outcomes:
        raise ValueError("majority of zero samples")
    value, count = Counter(outcomes).most_common(1)[0]
    return value if count * 2 > len(outcomes) else UNSTABLE


# --- results -----------------------------------------------------------------


@dataclass
class ProbeRun:
    """k samples of one probe context.

    raw: final response per sample (after the retry, if one happened).
    parsed: validated Answer per sample, None when rejected twice.
    """

    raw: list[str]
    parsed: list[Answer | None]
    outcomes: list[str]
    majority: str
    rejections: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "parsed": [p.to_json() if p else None for p in self.parsed],
            "outcomes": self.outcomes,
            "majority": self.majority,
            "rejections": self.rejections,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ProbeRun:
        return cls(
            raw=d["raw"],
            parsed=[Answer.from_json(p) if p else None for p in d["parsed"]],
            outcomes=d["outcomes"],
            majority=d["majority"],
            rejections=d["rejections"],
        )


@dataclass
class Result:
    """One audited record. Status is `ok`, `non_compliant`, or `unstable_original`.

    probes: keyed by PROBES. loo: leave-one-out runs keyed by the removed unit id.
    verdict, flags, alignment are None unless status is `ok`.
    counterfactual: probe C, None when C was not run. Otherwise
        {"applicable": bool, "reason": str | None, "replacement": str | None,
         "target": {"unit", "start", "end", "text"} | None, "seed": int,
         "run": ProbeRun JSON | None, "verdict": "grounded" | "decorative" |
         "unstable" | None}; reason is set exactly when applicable is false. target is
        the span C edited in the original record (probes.counterfactual_plan).
    compliance: {"calls": int, "retries": int, "final_rejections": int,
                 "code": str | None}  (code set when status is non_compliant).
    """

    record: Record
    model: str
    k: int
    temperature: float | None
    status: str
    A: Answer | None = None
    supplied: bool = False
    probes: dict[str, ProbeRun] = field(default_factory=dict)
    loo: dict[str, ProbeRun] = field(default_factory=dict)
    verdict: str | None = None
    flags: dict[str, Any] | None = None
    alignment: dict[str, Any] | None = None
    compliance: dict[str, Any] = field(default_factory=dict)
    counterfactual: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "record": self.record.to_json(),
            "model": self.model,
            "k": self.k,
            "temperature": self.temperature,
            "status": self.status,
            "A": self.A.to_json() if self.A else None,
            "supplied": self.supplied,
            "probes": {name: p.to_json() for name, p in self.probes.items()},
            "loo": {uid: p.to_json() for uid, p in self.loo.items()},
            "verdict": self.verdict,
            "flags": self.flags,
            "alignment": self.alignment,
            "compliance": self.compliance,
            "counterfactual": self.counterfactual,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Result:
        return cls(
            record=Record.from_json(d["record"]),
            model=d["model"],
            k=d["k"],
            temperature=d["temperature"],
            status=d["status"],
            A=Answer.from_json(d["A"]) if d["A"] else None,
            supplied=d["supplied"],
            probes={n: ProbeRun.from_json(p) for n, p in d["probes"].items()},
            loo={u: ProbeRun.from_json(p) for u, p in d["loo"].items()},
            verdict=d["verdict"],
            flags=d["flags"],
            alignment=d["alignment"],
            compliance=d["compliance"],
            counterfactual=d.get("counterfactual"),  # absent in v1 results
        )


def read_results(path: str | Path) -> list[Result]:
    return [Result.from_json(d) for d in read_jsonl(path)]
