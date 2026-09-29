"""Model backends: the local Claude Code CLI, and a deterministic fake for offline runs."""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Protocol

from poindexter.contract import MAX_ANSWER_CHARS


class Backend(Protocol):
    model: str

    async def complete(
        self, system: str, user: str, temperature: float | None, sample_index: int
    ) -> str: ...


_RETRYABLE = re.compile(r"rate.?limit|overloaded|\b429\b|\b529\b", re.IGNORECASE)


class ClaudeCLIBackend:
    """Calls `claude -p` with no tools, no MCP, no settings, thinking off.

    The CLI cannot set temperature, so every call samples at its default.
    `sample_index` is not sent; it exists so the cache can hold k distinct samples.
    """

    def __init__(
        self,
        model: str,
        executable: str = "claude",
        tries: int = 5,
        backoff: float = 2.0,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.executable = executable
        self.tries = tries
        self.backoff = backoff
        self.timeout = timeout
        # A fixed empty directory, so the CLI finds no project CLAUDE.md or settings.
        self.cwd = Path(tempfile.gettempdir()) / "poindexter-empty-cwd"
        self.cwd.mkdir(exist_ok=True)

    def argv(self, system: str, user: str) -> list[str]:
        return [
            self.executable, "-p",
            "--model", self.model,
            "--system-prompt", system,
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "",
            "--no-session-persistence",
            "--settings", '{"alwaysThinkingEnabled":false}',
            "--output-format", "json",
            user,
        ]  # fmt: skip

    async def complete(
        self, system: str, user: str, temperature: float | None, sample_index: int
    ) -> str:
        if temperature is not None:
            raise ValueError("the Claude CLI cannot set temperature; pass None")
        for attempt in range(self.tries):
            ok, text = await self._once(system, user)
            if ok:
                return text
            if not _RETRYABLE.search(text) or attempt == self.tries - 1:
                break
            await asyncio.sleep(self.backoff * 2**attempt)
        raise RuntimeError(f"claude CLI failed ({self.model}): {text}")

    async def _once(self, system: str, user: str) -> tuple[bool, str]:
        """One CLI call. Returns (True, result text), or (False, the error text from the
        CLI's JSON output). Retry decisions look only at that JSON text, never at stderr.
        Output that is not a JSON object raises at once, with stderr for diagnosis."""
        proc = await asyncio.create_subprocess_exec(
            *self.argv(system, user),
            cwd=self.cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), self.timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"claude CLI timed out after {self.timeout}s") from None
        stdout, stderr = out.decode(), err.decode()
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            raise RuntimeError(
                f"claude CLI exit {proc.returncode}, output is not a JSON object; "
                f"stderr: {stderr.strip()!r}; stdout: {stdout.strip()[:500]!r}"
            )
        result = data.get("result")
        if proc.returncode == 0 and data.get("is_error") is False and isinstance(result, str):
            return True, result
        fields = [f"{k}: {data[k]}" for k in ("subtype", "result", "error") if data.get(k)]
        return False, f"exit {proc.returncode}, is_error {data.get('is_error')}: " + "; ".join(
            fields
        )


class FakeBackend:
    """Deterministic function of the prompt text, for offline runs and tests.

    Answers with the text of the first context unit that contains the question's last
    word and cites that unit; abstains if no unit does. `calls` counts invocations.
    """

    _UNIT = re.compile(r"^\[([^\]]+)\] (.*)$")

    def __init__(self, model: str = "fake") -> None:
        self.model = model
        self.calls = 0

    async def complete(
        self, system: str, user: str, temperature: float | None, sample_index: int
    ) -> str:
        self.calls += 1
        units, question = [], ""
        for line in user.splitlines():
            if m := self._UNIT.match(line):
                units.append((m.group(1), m.group(2)))
            elif line.startswith("Question: "):
                question = line.removeprefix("Question: ")
        words = re.findall(r"\w+", question.lower())
        keyword = words[-1] if words else None
        for uid, text in units:
            if keyword and keyword in text.lower():
                return json.dumps(
                    {"answer": text[:MAX_ANSWER_CHARS], "cites": [uid], "abstain": False}
                )
        return json.dumps({"answer": None, "cites": [], "abstain": True})


def make_backend(name: str, model: str | None) -> Backend:
    if name == "fake":
        return FakeBackend(model or "fake")
    if name == "claude":
        if not model:
            raise ValueError("--backend claude needs --model")
        return ClaudeCLIBackend(model)
    raise ValueError(f"unknown backend {name!r}")
