"""
Shared pytest fixtures for yang4owl/tests/.

Inserts the parent directory (yang4owl/) into sys.path so `import align_lexicons`
resolves regardless of the working directory pytest was invoked from -- required
because align_lexicons.py lives in a package-less sibling directory to
draft_lexicon.py / yang4owl.py, with no __init__.py.

conftest.py deliberately does NOT import align_lexicons at module level: Task 2
of the phase-01 plan is what creates yang4owl/align_lexicons.py, and importing
it lazily (inside each fixture body, only when a test actually uses the
fixture) keeps this harness independently verifiable before that module
exists.

Both client doubles below are substituted at the network boundary only --
everything else in a run (rdflib parsing, evidence normalization, label
scoring, transcript printing) executes for real against the real,
un-curated `lexicon/` directory (CONTEXT.md D-01). No test using either
fixture requires ANTHROPIC_API_KEY.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def lexicon_dir() -> Path:
    """The real, un-curated lexicon directory -- tests read real files,
    never synthetic Turtle, per CONTEXT.md D-01."""
    return Path(__file__).resolve().parents[1] / "lexicon"


class _ParsedResponse:
    """Mimics the return shape of anthropic's client.messages.parse(): the
    one attribute align_lexicons.py reads off it is .parsed_output."""

    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _RecordingMessages:
    """Stand-in for anthropic.Anthropic().messages -- the one seam
    align_lexicons.py touches. Records every call's kwargs and returns a
    scripted verdict resolved by the caller-supplied lookup function."""

    def __init__(self, verdict_for_call):
        self._verdict_for_call = verdict_for_call
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _ParsedResponse(self._verdict_for_call(kwargs))


class _RecordingClient:
    """Stand-in for anthropic.Anthropic() -- shape matches the real client
    at the one seam align_lexicons.py touches: client.messages.parse(...)."""

    def __init__(self, verdict_for_call):
        self.messages = _RecordingMessages(verdict_for_call)

    @property
    def calls(self):
        """Convenience alias for self.messages.calls -- lets a test assert
        both *that* the confirmation stage ran and *what text* it received
        without reaching through .messages each time."""
        return self.messages.calls


# Module-level default verdict kwargs for recording_client -- constructed into
# a real align_lexicons.MatchVerdict lazily inside the fixture body, since
# align_lexicons cannot be imported at conftest module level (see docstring).
_DEFAULT_VERDICT_KWARGS = {
    "verdict": "reject",
    "rationale": "Default recording_client verdict -- no scripted response configured.",
    "evidence_quote": "",
}


@pytest.fixture
def recording_client():
    """A recording Anthropic client double returning a module-level default
    MatchVerdict for every call, and appending each call's kwargs to a
    public .calls list."""
    import align_lexicons  # lazy import -- see module docstring

    default_verdict = align_lexicons.MatchVerdict(**_DEFAULT_VERDICT_KWARGS)
    return _RecordingClient(lambda kwargs: default_verdict)


def _extract_prompt_text(kwargs: dict) -> str:
    """Flatten every string found in a messages.parse(**kwargs) call into one
    haystack for substring lookups, so scripted_client doesn't need to know
    the exact request shape align_lexicons.py builds (system blocks vs.
    leading user message -- see RESEARCH.md Assumption A2)."""
    chunks = []

    def _walk(value):
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _walk(v)

    _walk(kwargs)
    return "\n".join(chunks)


@pytest.fixture
def scripted_client():
    """Factory fixture: given a mapping from (tapi_lex_id, ietf_lex_id) to a
    MatchVerdict, returns a recording client that looks up the verdict for
    the pair named in the call's prompt text (both lex ids must appear as
    substrings), falling back to a reject verdict for unmapped pairs."""
    import align_lexicons  # lazy import -- see module docstring

    def _make(verdicts_by_pair: dict):
        fallback = align_lexicons.MatchVerdict(
            verdict="reject",
            rationale="scripted_client fallback -- no verdict mapped for this pair.",
            evidence_quote="",
        )

        def _verdict_for_call(kwargs):
            prompt_text = _extract_prompt_text(kwargs)
            for (tapi_lex_id, ietf_lex_id), verdict in verdicts_by_pair.items():
                if tapi_lex_id in prompt_text and ietf_lex_id in prompt_text:
                    return verdict
            return fallback

        return _RecordingClient(_verdict_for_call)

    return _make
