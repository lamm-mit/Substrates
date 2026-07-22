from __future__ import annotations

import analyze


def test_analysis_payload_includes_controls_and_matched_logit_rank(tmp_path):
    rec = {
        "slug": "item",
        "title": "Item",
        "shape": "MULTIHOP",
        "domain": "materials",
        "description": "Description",
        "note": "Rationale",
        "prompt_text": "Question is",
        "band": [38, 92],
        "protocol": "lens_eval",
        "valid_for_metrics": False,
        "excluded_reasons": ["clean model did not produce the expected answer"],
        "baseline": {
            "required": True,
            "expected": ["answer"],
            "greedy_token": "other",
            "correct": False,
        },
        "generated_completion": " other",
        "emergence": [{
            "label": "latent",
            "best_rank": 4,
            "best_depth": 60,
            "onset_depth": None,
            "reached_top1": False,
            "logit_lens_best_rank": 99,
        }],
        "tracked_dropped": [],
        "layer_readouts": [],
        "surprising": [],
        "figures": {},
    }

    content = analyze._prompt_user_content(rec, tmp_path)
    payload = content[0]["text"]

    assert "Registered metric status: excluded" in payload
    assert "model greedy token: 'other'" in payload
    assert "registered correct: False" in payload
    assert "exclusion reason(s): clean model did not produce" in payload
    assert "J-lens rank 5" in payload
    assert "logit-lens rank 100" in payload


def test_analysis_payload_lists_every_verbal_report_trial(tmp_path):
    rec = {
        "slug": "report",
        "title": "Report",
        "shape": "REPORT_SWAP",
        "domain": "materials",
        "description": "Description",
        "note": "Rationale",
        "prompt_text": "Think of a defect.",
        "band": [38, 92],
        "protocol": "verbal_report",
        "valid_for_metrics": True,
        "excluded_reasons": [],
        "baseline": {"required": False},
        "generated_completion": "Vacancy",
        "emergence": [],
        "tracked_dropped": [],
        "layer_readouts": [],
        "surprising": [],
        "figures": {},
        "swap": {
            "protocol": "verbal_report",
            "clean_source": "Vacancy",
            "protocol_success_rate": 0.0,
            "trials": [
                {
                    "target": "vacancy",
                    "source_rank_clean": 0,
                    "source_rank_swapped": 9,
                    "target_rank_clean": 10,
                    "target_rank_swapped": 20,
                    "swapped_top": [["stop", 0.9]],
                    "protocol_success": False,
                },
                {
                    "target": "interstitial",
                    "source_rank_clean": 0,
                    "source_rank_swapped": 30,
                    "target_rank_clean": 100,
                    "target_rank_swapped": 4,
                    "swapped_top": [["Interstitial", 0.8]],
                    "protocol_success": True,
                },
            ],
        },
    }

    payload = analyze._prompt_user_content(rec, tmp_path)[0]["text"]

    assert "target 'vacancy' [case-only self-target" in payload
    assert "target 'interstitial'" in payload
    assert "swapped top-1 'Interstitial'" in payload
