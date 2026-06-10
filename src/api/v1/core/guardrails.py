"""
src/core/guardrails.py

Guardrails layer for the Credit Card Spend Summarizer API.

Validator from the Guardrails AI Hub (https://hub.guardrailsai.com):

  1. Toxicity check  — ToxicLanguage   applied to the QUERY   (input guard)

Install once before running:

  pip install guardrails-ai
  guardrails configure                                   # paste your hub token
  guardrails hub install hub://guardrails/toxic_language

Set GUARDRAILS_API_KEY in .env to configure the Hub token automatically.
Set GUARDRAILS_USE_REMOTE_INFERENCING=true to skip local model downloads.
"""

import os
import uuid
from dotenv import load_dotenv

load_dotenv(override=True)

# The ValidationError import path has shifted across guardrails versions — be
# defensive so this module imports cleanly regardless of the installed version.
try:
    from guardrails.errors import ValidationError
except Exception:  # pragma: no cover - import path varies by version
    ValidationError = Exception


# ── Configuration ─────────────────────────────────────────────────────────────

TOXICITY_THRESHOLD = float(os.getenv("GUARDRAIL_TOXICITY_THRESHOLD", "0.5"))


class GuardrailViolation(Exception):
    """Raised when an input guardrail blocks a request.

    guard   — short name of the guard that fired.
    message — user-facing explanation suitable for an HTTP 400 response body.
    """

    def __init__(self, guard: str, message: str):
        self.guard = guard
        self.message = message
        super().__init__(f"[{guard}] {message}")


# ── Lazy guard construction ────────────────────────────────────────────────────
# Building a Guard imports hub validators (and downloads their models on first
# use). We build lazily and cache, so importing this module never fails just
# because the validators are not installed yet — the clear error only surfaces
# when a guard is actually used.

_guards = None


def _ensure_guardrails_configured() -> None:
    """Write ~/.guardrailsrc from GUARDRAILS_API_KEY env var if not already present.

    Allows token configuration via .env instead of interactive
    `guardrails configure`. If the rc file already exists it is left untouched.

    Set GUARDRAILS_USE_REMOTE_INFERENCING=true to run validators on the
    Guardrails hosted endpoint (skips local model downloads).
    """
    api_key = os.getenv("GUARDRAILS_API_KEY")
    if not api_key:
        return

    os.environ.setdefault("GUARDRAILS_TOKEN", api_key)

    rc_path = os.path.expanduser("~/.guardrailsrc")
    if os.path.exists(rc_path):
        return

    use_remote = os.getenv("GUARDRAILS_USE_REMOTE_INFERENCING", "false")
    try:
        with open(rc_path, "w") as rc_file:
            rc_file.write(
                f"id={uuid.uuid4()}\n"
                f"token={api_key}\n"
                "enable_metrics=false\n"
                f"use_remote_inferencing={use_remote}\n"
            )
    except OSError:
        pass


def _build_guards() -> dict:
    _ensure_guardrails_configured()
    try:
        from guardrails import Guard
        from guardrails.hub import ToxicLanguage
    except ImportError as exc:
        raise RuntimeError(
            "Guardrails validators are not installed. Run:\n"
            "  pip install guardrails-ai\n"
            "  guardrails configure\n"
            "  guardrails hub install hub://guardrails/toxic_language"
        ) from exc

    return {
        # Input guard — raise if the query is toxic.
        "toxicity": Guard().use(
            ToxicLanguage(
                threshold=TOXICITY_THRESHOLD,
                validation_method="sentence",
                on_fail="exception",
            )
        ),
    }


def _get_guards() -> dict:
    global _guards
    if _guards is None:
        _guards = _build_guards()
    return _guards


# ── Public API ─────────────────────────────────────────────────────────────────


def guard_input(query: str) -> None:
    """Run input guardrails on the user's query.

    Raises GuardrailViolation if the query is flagged as toxic.
    Call this before passing the query to the agent graph.
    """
    guards = _get_guards()
    try:
        guards["toxicity"].validate(query)
    except ValidationError as exc:
        raise GuardrailViolation(
            "toxic_language",
            "Your message was flagged as abusive or toxic and cannot be processed.",
        ) from exc