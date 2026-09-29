import json

import pytest


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own cache file, so no test sees another's responses."""
    path = tmp_path / "cache.sqlite"
    monkeypatch.setenv("POINDEXTER_CACHE", str(path))
    return path


def answer(text, *cites):
    return json.dumps({"answer": text, "cites": list(cites), "abstain": False})


ABSTAIN_JSON = json.dumps({"answer": None, "cites": [], "abstain": True})


class ScriptedBackend:
    """Canned responses chosen by prompt substring.

    rules: (substring, response) pairs, first match on the user prompt wins. A response
    is a string, or a list indexed by sample_index. No match is a test bug and raises.
    """

    def __init__(self, rules, model="scripted"):
        self.rules = rules
        self.model = model
        self.calls = 0
        self.prompts = []
        self.systems = []

    async def complete(self, system, user, temperature, sample_index):
        self.calls += 1
        self.prompts.append(user)
        self.systems.append(system)
        for needle, response in self.rules:
            if needle in user:
                return response if isinstance(response, str) else response[sample_index]
        raise AssertionError(f"no scripted response for prompt:\n{user}")
