import asyncio
import json
import os
import stat
from pathlib import Path

import pytest
from conftest import ABSTAIN_JSON, ScriptedBackend, answer

from poindexter import cache
from poindexter.backend import ClaudeCLIBackend, FakeBackend
from poindexter.contract import (
    ABSTAIN,
    OTHER,
    SAME,
    Answer,
    Record,
    Unit,
    read_records,
    read_results,
)
from poindexter.prompt import (
    CLOSED_BOOK_SYSTEM,
    SYSTEM,
    build_closed_book_prompt,
    build_prompt,
    validate_closed_book,
)
from poindexter.runner import closed_book, run_file, run_probe, run_record, run_records
from poindexter.verdict import recompute

TOY = Path(__file__).parent.parent / "examples" / "toy.jsonl"

UNITS = [Unit("u1", "The store opened in 1998."), Unit("u2", "The warranty lasts 90 days.")]
Q = "How long is the warranty?"
CB = build_closed_book_prompt(Q)[1]
RECORD = Record("r1", Q, UNITS)


def user(units):
    return build_prompt(units, Q)[1]


def go(coro):
    return asyncio.run(coro)


def record_run(backend, record=RECORD, k=3, probes="verdict"):
    return go(run_record(record, backend, k, None, 0, None, asyncio.Semaphore(4), probes))


def test_toy_end_to_end(tmp_path):
    fake = FakeBackend()
    out = tmp_path / "r.jsonl"
    results = go(run_file(TOY, fake, 3, None, 0, 8, out))
    by_id = {r.record.id: r for r in read_results(out)}
    assert len(by_id) == len(results) == 6
    assert {i: r.verdict for i, r in by_id.items()} == {
        "toy-grounded": "grounded",
        "toy-redundant": "decorative",
        "toy-fabricates": "grounded",
        "toy-abstain": None,
        "toy-supplied": "grounded",
        "toy-incomplete": "incomplete",
    }
    assert by_id["toy-fabricates"].flags["fabricates"]
    assert by_id["toy-abstain"].A.abstain and list(by_id["toy-abstain"].probes) == ["O"]
    assert by_id["toy-supplied"].supplied and by_id["toy-supplied"].flags["reproduced"]
    assert by_id["toy-incomplete"].flags["reproduced"] is False
    assert by_id["toy-grounded"].alignment == {
        "precision": 1.0, "recall": 1.0, "load_bearing": ["u2"],
    }  # fmt: skip
    assert by_id["toy-redundant"].alignment["recall"] is None
    grounded = by_id["toy-grounded"]
    assert list(grounded.probes) == ["O", "N", "R_remove", "R_replace", "M", "S"]
    assert list(grounded.loo) == ["u1", "u2", "u3"]
    assert all(len(p.raw) == 3 for p in grounded.probes.values())
    assert grounded.temperature is None

    calls = fake.calls
    assert calls > 0
    go(run_file(TOY, fake, 3, None, 0, 8, tmp_path / "again.jsonl"))
    assert fake.calls == calls  # second run is served entirely from the cache


def test_verdict_probe_set(tmp_path):
    results = go(run_file(TOY, FakeBackend(), 1, None, 0, 8, None, probes="verdict"))
    r = next(r for r in results if r.record.id == "toy-grounded")
    assert list(r.probes) == ["O", "N", "R_remove", "M"]
    assert r.loo == {} and r.alignment is None
    assert r.flags["fabricates_on_replace"] is None and r.flags["position_sensitive"] is None
    assert r.verdict == "grounded"


def test_all_probes_need_two_records():
    with pytest.raises(ValueError):
        go(run_file(TOY, FakeBackend(), 1, None, 0, 8, None, probes="bogus"))
    one = read_records(TOY)[:1]
    with pytest.raises(ValueError):
        go(run_records(one, FakeBackend(), 1, None, 0))


def test_generated_answer_takes_first_majority_sample():
    b = ScriptedBackend([
        (user(UNITS), [answer("30 days", "u1"), answer("90 days", "u2"), answer("90 Days", "u1")]),
        (user(UNITS[:1]), ABSTAIN_JSON),
        (user(UNITS[1:]), answer("90 days", "u2")),
        (CB, ABSTAIN_JSON),
    ])  # fmt: skip
    r = record_run(b)
    assert r.status == "ok"
    assert r.A == Answer("90 days", ["u2"], False)
    assert r.probes["O"].outcomes == [OTHER, SAME, SAME]
    assert r.verdict == "grounded"


def test_retry_then_double_rejection_counts_as_other():
    retry_tail = "Your previous response was rejected: INVALID_JSON"
    b = ScriptedBackend([
        (retry_tail, ["still not json", answer("90 days", "u2"), "nope"]),
        (user(UNITS), ["not json", "not json", answer("90 days", "u2")]),
        (user(UNITS[:1]), ABSTAIN_JSON),
        (user(UNITS[1:]), answer("90 days", "u2")),
        (CB, ABSTAIN_JSON),
    ])  # fmt: skip
    r = record_run(b)
    o = r.probes["O"]
    assert o.parsed[0] is None and o.raw[0] == "still not json"
    assert o.outcomes == [OTHER, SAME, SAME] and o.rejections == 1
    retries = [p for p in b.prompts if retry_tail in p]
    assert len(retries) == 2
    assert retries[0].endswith(". Respond with only the JSON object.")
    assert r.compliance["retries"] == 2 and r.compliance["final_rejections"] == 1


def test_non_compliant_and_unstable_original():
    b = ScriptedBackend([("", '{"answer": "x"}')])
    r = record_run(b)
    assert r.status == "non_compliant" and r.compliance["code"] == "BAD_KEYS"
    assert r.A is None and r.verdict is None and list(r.probes) == ["O"]
    b = ScriptedBackend([(user(UNITS), [answer("a", "u1"), answer("b", "u1")])], model="m2")
    r = record_run(b, k=2)
    assert r.status == "unstable_original" and r.compliance["code"] is None


def test_supplied_answer_is_used_and_reproduced_checked():
    supplied = Answer("ninety days", ["u2"], False)
    rec = Record("r1", Q, UNITS, answer=supplied)
    b = ScriptedBackend([
        (user(UNITS), answer("90 days", "u2")),
        (user(UNITS[:1]), ABSTAIN_JSON),
        (user(UNITS[1:]), answer("ninety days", "u2")),
        (CB, ABSTAIN_JSON),
    ])  # fmt: skip
    r = record_run(b, record=rec)
    assert r.A == supplied and r.supplied
    assert r.flags["reproduced"] is False
    assert r.verdict == "grounded"


def test_run_probe_validates_against_context_ids():
    a = Answer("90 days", ["u2"], False)
    b = ScriptedBackend([("Question", answer("90 days", "u2"))])
    run = go(run_probe(UNITS[:1], Q, a, b, 2, None, asyncio.Semaphore(2)))
    assert run.parsed == [None, None] and run.outcomes == [OTHER, OTHER] and run.rejections == 2
    run = go(run_probe(UNITS, Q, a, b, 2, None, asyncio.Semaphore(2)))
    assert run.majority == SAME


def test_closed_book():
    rec = Record("r", "When was it built?", [], meta={"dataset_answers": ["1889", "in 1889"]})
    b = ScriptedBackend([("", [
        json.dumps({"answer": "1889.", "cites": [], "abstain": False}),
        ABSTAIN_JSON,
        json.dumps({"answer": "1890", "cites": [], "abstain": False}),
    ])])  # fmt: skip
    run = go(closed_book(rec, b, 3, asyncio.Semaphore(2)))
    assert run.outcomes == [SAME, ABSTAIN, OTHER]
    assert b.prompts[0] == build_closed_book_prompt("When was it built?")[1]
    with pytest.raises(ValueError):
        go(closed_book(Record("r", "q", []), b, 1, asyncio.Semaphore(1)))


def test_validate_closed_book():
    ok = validate_closed_book(json.dumps({"answer": "1889", "cites": [], "abstain": False}))
    assert ok == Answer("1889", [], False)
    bad = validate_closed_book(json.dumps({"answer": "1889", "cites": ["u1"], "abstain": False}))
    assert bad.code == "BAD_CITES"
    assert validate_closed_book('{"answer": "x"}').code == "BAD_KEYS"
    assert validate_closed_book("nope").code == "INVALID_JSON"
    bad = validate_closed_book(json.dumps({"answer": "x", "cites": [], "abstain": True}))
    assert bad.code == "BAD_ABSTAIN"
    assert validate_closed_book(json.dumps({"answer": "", "cites": [], "abstain": False})).code == (
        "BAD_ANSWER"
    )


def test_cache_key_covers_every_field(isolated_cache):
    base = ("m", None, "sys", "usr", 0)
    k0 = cache.key(*base)
    for i, other in enumerate(["m2", 0.5, "sys2", "usr2", 1]):
        changed = list(base)
        changed[i] = other
        assert cache.key(*changed) != k0
    assert cache.key("m", 0.0, "s", "u", 0) != cache.key("m", None, "s", "u", 0)
    store = cache.open_default()
    assert store.path == isolated_cache.resolve()
    assert store.get(k0) is None
    assert store.put(k0, "first") == "first"
    assert store.put(k0, "second") == "first"  # first writer wins
    assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


# --- ClaudeCLIBackend against a stand-in executable ---------------------------

SCRIPT = """#!/bin/sh
n=$(cat "$FAKE_DIR/count" 2>/dev/null || echo 0)
n=$((n+1)); echo $n > "$FAKE_DIR/count"
pwd > "$FAKE_DIR/cwd"
for a in "$@"; do printf '%s\\0' "$a"; done > "$FAKE_DIR/argv"
if [ -n "$NOT_JSON" ]; then
  echo "429 rate limit" >&2; echo "oops"; exit 1
fi
if [ "$n" -le "$FAILS" ]; then
  echo '{"is_error": true, "result": "'"$ERROR"'"}'; exit 1
fi
echo '{"type": "result", "is_error": false, "result": "{\\"ok\\": true}"}'
"""


@pytest.fixture(scope="module")
def fake_cli_exe(tmp_path_factory):
    """Written once: macOS scans each new executable on its first run (~0.1 s)."""
    exe = tmp_path_factory.mktemp("cli") / "claude"
    exe.write_text(SCRIPT)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


@pytest.fixture
def fake_cli(fake_cli_exe, tmp_path, monkeypatch):
    exe = fake_cli_exe
    monkeypatch.setenv("FAKE_DIR", str(tmp_path))
    monkeypatch.delenv("NOT_JSON", raising=False)
    monkeypatch.setenv("FAILS", "0")
    monkeypatch.setenv("ERROR", "")
    return ClaudeCLIBackend("claude-test", executable=str(exe), backoff=0.0), tmp_path


def test_cli_backend_argv_and_result(fake_cli):
    backend, d = fake_cli
    assert go(backend.complete("SYS", "USER TEXT", None, 0)) == '{"ok": true}'
    argv = (d / "argv").read_text().split("\0")[:-1]
    assert argv == [
        "-p", "--model", "claude-test", "--system-prompt", "SYS", "--tools", "",
        "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
        "--settings", '{"alwaysThinkingEnabled":false}', "--output-format", "json",
        "USER TEXT",
    ]  # fmt: skip
    cwd = Path((d / "cwd").read_text().strip())
    assert cwd.resolve() == backend.cwd.resolve() and os.listdir(cwd) == []


def test_cli_backend_retries_overload_then_succeeds(fake_cli, monkeypatch):
    backend, d = fake_cli
    monkeypatch.setenv("FAILS", "2")
    monkeypatch.setenv("ERROR", "API Error: 529 Overloaded")
    assert go(backend.complete("S", "U", None, 0)) == '{"ok": true}'
    assert (d / "count").read_text().strip() == "3"


def test_cli_backend_gives_up_after_five_tries(fake_cli, monkeypatch):
    backend, d = fake_cli
    monkeypatch.setenv("FAILS", "9")
    monkeypatch.setenv("ERROR", "rate limit exceeded")
    with pytest.raises(RuntimeError, match="rate limit"):
        go(backend.complete("S", "U", None, 0))
    assert (d / "count").read_text().strip() == "5"


def test_cli_backend_raises_on_other_errors(fake_cli, monkeypatch):
    backend, d = fake_cli
    monkeypatch.setenv("FAILS", "9")
    monkeypatch.setenv("ERROR", "invalid model")
    with pytest.raises(RuntimeError, match="invalid model"):
        go(backend.complete("S", "U", None, 0))
    assert (d / "count").read_text().strip() == "1"


def test_cli_backend_rejects_temperature(fake_cli):
    backend, d = fake_cli
    with pytest.raises(ValueError):
        go(backend.complete("S", "U", 0.0, 0))
    assert not (d / "count").exists()


# --- probe N: closed-book prompt ---------------------------------------------


def cb(text):
    return json.dumps({"answer": text, "cites": [], "abstain": False})


def test_parametric_end_to_end():
    """The model answers the same from memory under N: decorative plus parametric."""
    b = ScriptedBackend([
        (user(UNITS), answer("90 days", "u2")),
        (user(UNITS[:1]), answer("90 days", "u1")),
        (user(UNITS[1:]), answer("90 days", "u2")),
        (CB, [cb("90 Days"), cb("90 days"), ABSTAIN_JSON]),
    ])  # fmt: skip
    r = record_run(b)
    n = r.probes["N"]
    assert n.outcomes == [SAME, SAME, ABSTAIN] and n.majority == SAME
    assert r.flags["parametric"] is True
    assert r.verdict == "decorative"
    assert CB in b.prompts and not any("(no context units)" in p for p in b.prompts)


def test_n_uses_closed_book_validator_with_one_retry():
    """N rejects cites (cites must be []), retries once, then counts the sample as other."""
    b = ScriptedBackend([
        ("rejected: BAD_CITES", [cb("90 days"), answer("90 days", "u2"), cb("30 days")]),
        (user(UNITS), answer("90 days", "u2")),
        (user(UNITS[:1]), ABSTAIN_JSON),
        (user(UNITS[1:]), answer("90 days", "u2")),
        (CB, answer("90 days", "u2")),
    ])  # fmt: skip
    r = record_run(b)
    n = r.probes["N"]
    assert n.outcomes == [SAME, OTHER, OTHER] and n.rejections == 1
    assert r.flags["parametric"] is False
    assert r.compliance["retries"] == 3


def test_n_is_cached_apart_from_context_prompts():
    k_n = cache.key("m", None, CLOSED_BOOK_SYSTEM, CB, 0)
    k_ctx = cache.key("m", None, SYSTEM, build_prompt([], Q)[1], 0)
    assert k_n != k_ctx


def test_o_only_probe_set():
    b = ScriptedBackend([(user(UNITS), [answer("90 days", "u2"), answer("30 days", "u1"),
                                        answer("90 days", "u1")])])  # fmt: skip
    r = record_run(b, probes="O")
    assert r.status == "ok" and r.A == Answer("90 days", ["u2"], False)
    assert list(r.probes) == ["O"] and r.probes["O"].outcomes == [SAME, OTHER, SAME]
    assert r.verdict is None and r.flags is None and r.alignment is None and r.loo == {}
    assert b.calls == 3 and r.compliance["calls"] == 3
    assert recompute(r, 1).verdict is None
    results = go(run_file(TOY, FakeBackend(), 1, None, 0, 8, None, probes="O"))
    assert all(list(x.probes) == ["O"] and x.verdict is None for x in results)


def test_pick_answer_groups_by_canonical_text():
    b = ScriptedBackend([(user(UNITS), [answer("ninety", "u1"), answer("in 90", "u2"),
                                        answer("90 days", "u2")])])  # fmt: skip
    r = record_run(b, probes="O")  # "ninety" is past twenty, so it is plain text
    assert r.status == "ok" and r.A == Answer("in 90", ["u2"], False)
    assert r.probes["O"].outcomes == [OTHER, SAME, SAME]


def test_closed_book_uses_canonical():
    rec = Record("r", "How many?", [], meta={"dataset_answers": ["2"]})
    b = ScriptedBackend([("", json.dumps({"answer": "two atoms", "cites": [], "abstain": False}))])
    assert go(closed_book(rec, b, 1, asyncio.Semaphore(1))).outcomes == [SAME]


def test_cli_backend_non_json_output_raises_at_once(fake_cli, monkeypatch):
    backend, d = fake_cli
    monkeypatch.setenv("NOT_JSON", "1")
    with pytest.raises(RuntimeError, match="not a JSON object.*429 rate limit"):
        go(backend.complete("S", "U", None, 0))
    assert (d / "count").read_text().strip() == "1"  # stderr text never triggers a retry
