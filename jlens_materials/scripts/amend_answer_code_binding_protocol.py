#!/usr/bin/env python3
"""Record the pre-forward contextual-tokenization amendment, exactly once."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import freeze_lexical_adversarial_representation as shared

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "answer-code-binding-2026-07-17"
ORIGINAL = OUT / "protocol.json"
AMENDED = OUT / "protocol-amendment-v1.json"
RUNNER = ROOT / "scripts" / "run_answer_code_binding.py"
ATTEMPT = OUT / "execution-attempt-1-pre-forward.json"


def main() -> None:
    if AMENDED.exists() or ATTEMPT.exists():
        raise FileExistsError("Amendment record already exists; do not overwrite.")
    protocol = json.loads(ORIGINAL.read_text())
    original_hash = shared.sha256(ORIGINAL)
    original_runner_hash = protocol["inputs"]["runner_sha256"]
    protocol["status"] = (
        "prospectively frozen scientific design; protocol amendment v1 was "
        "made after execution attempt 1 aborted before any model forward pass"
    )
    protocol["amendment"] = {
        "amended_at": datetime.now(timezone.utc).isoformat(),
        "original_protocol": str(ORIGINAL.relative_to(ROOT)),
        "original_protocol_sha256": original_hash,
        "reason": (
            "The isolated token `checkpoint` has id 73093, whereas the exact "
            "in-prompt substring ` checkpoint` has id 61077. Attempt 1 searched "
            "for 73093, found zero matches, and raised before model.forward."
        ),
        "scientific_design_changed": False,
        "prompts_changed": False,
        "endpoints_changed": False,
        "only_change": (
            "Match the already-frozen marker in its exact contextual "
            "tokenization, ` checkpoint`."
        ),
        "attempt_record": str(ATTEMPT.relative_to(ROOT)),
    }
    protocol["checkpoint_token_text"] = " checkpoint"
    protocol["tokenization_preflight"][" checkpoint"] = [61077]
    protocol["inputs"]["runner_sha256_before_amendment"] = original_runner_hash
    protocol["inputs"]["runner_sha256"] = shared.sha256(RUNNER)
    shared.dump(AMENDED, protocol)
    shared.dump(ATTEMPT, {
        "study_id": protocol["study_id"],
        "attempt": 1,
        "outcome": "aborted before any model forward pass",
        "original_protocol_sha256": original_hash,
        "original_runner_sha256": original_runner_hash,
        "model_loaded": True,
        "n_prompts_forwarded": 0,
        "error": (
            "First prompt had zero matches for isolated checkpoint token id "
            "73093; contextual token id is 61077."
        ),
        "retained_because": (
            "Pre-forward failures and amendments are part of the complete "
            "prospective record."
        ),
    })
    print(f"wrote {AMENDED.relative_to(ROOT)}")
    print(f"amended protocol sha256: {shared.sha256(AMENDED)}")


if __name__ == "__main__":
    main()
