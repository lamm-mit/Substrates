from __future__ import annotations

import json

import report
import report_latex


def _run_record():
    metric = {
        "n_items": 50,
        "minimum_required": 50,
        "sufficient_sample": True,
        "ks": [1, 5],
        "jacobian_lens": {"pass_at_k": [0.2, 0.8], "auc_log_k": 0.5},
        "logit_lens": {"pass_at_k": [0.1, 0.4], "auc_log_k": 0.25},
    }
    return {
        "format_version": 2,
        "model": "test/model",
        "tag": "test",
        "n_layers": 10,
        "d_model": 16,
        "lens_n_prompts": 1000,
        "shapes": ["MULTIHOP"],
        "domains": ["mechanics"],
        "methodology": {
            "claims_level": "paper-protocol quantitative",
            "recipe": {"name": "paper"},
            "workspace_band": [38, 92],
        },
        "metrics": {
            "by_shape": {"MULTIHOP": metric},
            "directed_modulation_controls": {
                "group": {
                    "focus": {"n_items": 1, "hit_rate": 0.5,
                              "sufficient_sample": False},
                    "suppress": {"n_items": 1, "hit_rate": 0.0,
                                 "sufficient_sample": False},
                    "control": {"n_items": 1, "hit_rate": 0.0,
                                "sufficient_sample": False},
                    "summary": {"distinct_phrasings": 3,
                                "paper_phrasing_target": 24},
                }
            },
            "causal_swaps": {
                "n_interventions": 2,
                "n_graded": 2,
                "minimum_required": 50,
                "sufficient_sample": False,
                "protocol_success_rate": 0.5,
                "n_counterfactual": 1,
                "causal_success_rate": 0.0,
            },
        },
        "prompts": [{
            "slug": "item",
            "shape": "MULTIHOP",
            "domain": "mechanics",
            "title": "Item",
            "description": "Description",
            "prompt_text": "Question is",
            "valid_for_metrics": True,
            "excluded_reasons": [],
            "baseline": {
                "required": True, "correct": True,
                "expected": ["answer"], "greedy_token": "answer",
            },
            "generated_completion": "\n\n",
            "tracked_dropped": [],
            "emergence": [{
                "label": "latent", "best_rank": 0, "best_depth": 60,
                "onset_depth": 40, "reached_top1": True,
                "logit_lens_best_rank": 7,
            }],
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
        }],
    }


def test_markdown_and_latex_reports_render_v2_methodology(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    path = runs / "test.json"
    path.write_text(json.dumps(_run_record()))
    markdown_path = report.build(str(path))
    latex_path = report_latex.build_tex(str(path))
    pdf_path = report_latex.compile_pdf(latex_path)
    markdown = markdown_path.read_text()
    latex = latex_path.read_text()
    assert "paper-protocol quantitative" in markdown
    assert "sustained onset" in markdown
    assert "Item-level lens evaluation" in markdown
    assert "paper-protocol quantitative" in latex
    assert "Item-level lens evaluation" in latex
    assert "estimated per prompt" not in markdown
    assert "estimated per prompt" not in latex
    assert "Fixed workspace band" in markdown
    assert "Fixed workspace band" in latex
    assert "logit-lens best rank" in markdown
    assert "logit rank" in latex
    assert "included in the registered aggregate" in markdown
    assert "Included in the registered aggregate" in latex
    assert "Clean baseline" in markdown
    assert "Clean baseline" in latex
    assert r"\n\n" in markdown
    assert '"[newline][newline]"' in latex
    assert "case-only self-target" in markdown
    assert "case-only self-target" in latex
    assert "interstitial" in markdown
    assert "interstitial" in latex
    assert "Directed-modulation controls" in markdown
    assert "Directed-modulation controls" in latex
    assert "Causal-intervention summary" in markdown
    assert "Causal-intervention summary" in latex
    if pdf_path is not None:
        assert pdf_path.is_file() and pdf_path.stat().st_size > 0


def test_latex_escape_handles_common_model_generated_unicode():
    escaped = report_latex.tex_escape("A → B ⇒ C × D — ‘quoted’")
    assert r"$\rightarrow$" in escaped
    assert r"$\Rightarrow$" in escaped
    assert r"$\times$" in escaped
    assert "---" in escaped
    assert not any(char in escaped for char in "→⇒×—‘’")


def test_markdown_conversion_never_leaks_sentinels_from_bold_inside_code():
    latex = report_latex.md_to_tex(
        "The token was `**not** 3,831 **and malformed.` Then **bold**."
    )

    assert "\x00" not in latex
    assert r"\texttt{" in latex
    assert r"\textbf{bold}" in latex


def test_legacy_reports_are_explicitly_qualitative(tmp_path):
    run = _run_record()
    run.pop("format_version")
    run.pop("methodology")
    run.pop("metrics")
    run["prompts"][0]["emergence"][0].pop("onset_depth")
    for key in ("valid_for_metrics", "excluded_reasons", "baseline",
                "generated_completion"):
        run["prompts"][0].pop(key)
    runs = tmp_path / "runs"
    runs.mkdir()
    path = runs / "legacy.json"
    path.write_text(json.dumps(run))

    markdown = report.build(str(path)).read_text()
    latex = report_latex.build_tex(str(path)).read_text()

    assert "Legacy exploratory run" in markdown
    assert "Legacy methodology" in markdown
    assert "Legacy exploratory run" in latex
    assert "Legacy methodology" in latex
    assert "estimated per prompt" not in markdown
    assert "estimated per prompt" not in latex
