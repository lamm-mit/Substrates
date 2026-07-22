from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import domain_prompts as dp
import matviz
import paper_protocol as pp
import run_lens


class WordTokenizer:
    all_special_ids = [0]
    unk_token_id = 99
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        words = [
            "alpha", "beta", "gamma", "answer", "The", "carrier", "sentence",
            "focus", "newline", "\n",
        ]
        self.vocab = {word: i + 1 for i, word in enumerate(words)}
        self.name_or_path = "word-tokenizer"

    def encode(self, text, add_special_tokens=True):
        pieces = text.replace("\n", " \n ").split()
        ids = [self.vocab.get(piece.strip(), self.unk_token_id) for piece in pieces]
        return ([0] + ids) if add_special_tokens else ids

    def decode(self, ids, **_):
        reverse = {value: key for key, value in self.vocab.items()}
        return " ".join(reverse.get(int(token_id), "<unk>") for token_id in ids)

    def __len__(self):
        return 100


def test_paper_recipe_and_evenly_spaced_layers():
    recipe = pp.resolve_recipe("paper")
    assert recipe.n_fit == 1000
    assert recipe.max_seq_len == 128
    assert recipe.target_layer == -2
    layers = pp.evenly_spaced_source_layers(64, recipe.target_layer, 25)
    assert len(layers) == 25
    assert layers[0] == 0
    assert layers[-1] == 61


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--allow-unverified-lens"], "exploratory only"),
        (["--no-strict-protocol"], "requires strict protocol"),
        (["--layer-stride", "2"], "requires --layer-stride 1"),
        (["--min-items-per-shape", "1"], "at least 50"),
    ],
)
def test_paper_cli_rejects_claim_downgrading_overrides(monkeypatch, extra, message):
    monkeypatch.setattr("sys.argv", ["run_lens.py", "--recipe", "paper", *extra])
    with pytest.raises(SystemExit, match=message):
        run_lens.main()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--hf-upload-lens"], "require --hf-lens-repo"),
        (["--hf-lens-repo", "owner/private"], "requires an explicit local --lens"),
    ],
)
def test_hub_cli_requires_repo_and_local_path(monkeypatch, args, message):
    monkeypatch.setattr("sys.argv", ["run_lens.py", *args])
    with pytest.raises(SystemExit, match=message):
        run_lens.main()


def test_builtin_corpus_never_cycles_records():
    with pytest.raises(ValueError, match="independent texts"):
        pp.load_fit_corpus(corpus="builtin", n=1000, strict=True)
    corpus = pp.load_fit_corpus(corpus="builtin", n=1000, strict=False)
    assert len(corpus.texts) == 8
    assert corpus.metadata["records"] == 8
    assert len(corpus.metadata["sha256"]) == 64


def test_lens_bundle_save_creates_explicit_parent_directory(tmp_path):
    class FakeLens:
        def save(self, path):
            Path(path).write_bytes(b"weights")

    destination = tmp_path / "nested" / "lens.pt"
    metadata_path = run_lens._save_lens_bundle(
        FakeLens(), destination, {"format_version": 1}
    )

    assert destination.read_bytes() == b"weights"
    assert json.loads(metadata_path.read_text()) == {"format_version": 1}


def test_file_corpus_is_content_addressed(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"texts": ["a" * 700, "b" * 700]}))
    first = pp.load_fit_corpus(corpus=str(path), n=2, min_chars=600, seed=7)
    second = pp.load_fit_corpus(corpus=str(path), n=2, min_chars=600, seed=7)
    assert first.texts == second.texts
    assert first.metadata["sha256"] == second.metadata["sha256"]


def test_strict_corpus_rejects_duplicate_records(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps({"texts": ["same " * 150, "same " * 150]}))
    with pytest.raises(ValueError, match="duplicate fitting records"):
        pp.load_fit_corpus(corpus=str(path), n=2, min_chars=600, strict=True)


def test_original_release_item_schema_maps_to_prompt():
    source = json.dumps({"items": [{
        "name": "mechanics-item",
        "prompt": "The computed driving value exceeds the critical value, so it will",
        "intermediate": "unstable",
        "answer": "grow",
        "swap_to": "stable",
        "swap_answer": "stop",
    }]})
    prompt = dp.load_prompts(source)[0]
    assert prompt.slug == "mechanics-item"
    assert prompt.protocol == "probe_swap"
    assert prompt.shape == "PROBE_SWAP"
    assert prompt.tracked == ("unstable",)
    assert prompt.readout_selector == "before_answer"
    assert prompt.must_be_absent_from_input


def test_release_filename_selects_its_fixed_scoring_rule(tmp_path):
    path = tmp_path / "lens-eval-association.json"
    path.write_text(json.dumps({"items": [{
        "name": "evoked", "prompt": "A quiet unnamed vignette.",
        "intermediates": ["grief"],
    }]}))
    prompt = dp.load_prompts(str(path))[0]
    assert prompt.shape == "ASSOCIATION"
    assert prompt.readout_selector == "final_prompt_token"
    assert prompt.must_be_absent_from_output


def test_mechanics_example_exercises_all_protocol_families():
    path = (Path(__file__).resolve().parents[1] / "jlens_materials" / "prompts" /
            "mechanics-paper-example.json")
    prompts = dp.load_prompts(str(path))
    assert {p.protocol for p in prompts} == {
        "lens_eval", "probe_swap", "directed_modulation", "verbal_report"
    }
    conditions = {p.condition for p in prompts if p.protocol == "directed_modulation"}
    assert conditions == {"focus", "suppress", "control"}


def test_materials_pack_is_small_controlled_case_study():
    path = (Path(__file__).resolve().parents[1] / "jlens_materials" / "prompts" /
            "materials-qualitative-pack.json")
    prompts = dp.load_prompts(str(path))
    assert len(prompts) == 14
    assert {p.protocol for p in prompts} == {
        "lens_eval", "probe_swap", "directed_modulation", "verbal_report"
    }
    assert {p.condition for p in prompts if p.protocol == "directed_modulation"} == {
        "focus", "suppress", "control"
    }
    lens_items = [p for p in prompts if p.protocol == "lens_eval"]
    assert lens_items
    assert all(p.must_be_absent_from_input and p.must_be_absent_from_output
               for p in lens_items)
    assert all(p.readout_selector != "all_prompt" for p in prompts)


def test_bundled_legacy_json_is_never_counted_as_paper_evaluation():
    path = (Path(__file__).resolve().parents[1] / "jlens_materials" / "prompts" /
            "fracture.json")
    prompts = dp.load_prompts(str(path))
    assert prompts
    assert {p.protocol for p in prompts} == {"exploratory"}


def test_synonyms_resolve_to_one_concept_with_multiple_ids():
    tokenizer = WordTokenizer()
    prompt = dp.Prompt(
        slug="syn", shape="MULTIHOP", domain="test", title="syn",
        description="", text="alpha", tracked=("alpha",),
        synonyms={"alpha": ["beta"]},
    )
    concepts, dropped = pp.resolve_concepts(tokenizer, prompt)
    assert not dropped
    assert len(concepts) == 1
    assert set(concepts[0].token_ids) == {tokenizer.vocab["alpha"], tokenizer.vocab["beta"]}


def test_fixed_readout_selectors_and_all_prompt_guard():
    tokenizer = WordTokenizer()
    prompt = dp.Prompt(
        slug="fixed", shape="MULTIHOP", domain="test", title="fixed",
        description="", text="alpha beta", readout_selector="before_answer",
    )
    assert pp.resolve_score_positions(tokenizer, prompt.text, prompt, strict=True) == [2]
    exploratory = dp.Prompt(
        slug="all", shape="MULTIHOP", domain="test", title="all",
        description="", text="alpha beta", readout_selector="all_prompt",
    )
    with pytest.raises(ValueError, match="exploratory"):
        pp.resolve_score_positions(tokenizer, exploratory.text, exploratory, strict=True)


def test_protocol_presence_uses_token_ids_not_substrings():
    tokenizer = WordTokenizer()
    prompt = dp.Prompt(
        slug="absence", shape="MULTIHOP", domain="test", title="absence",
        description="", text="beta", tracked=("alpha",),
        must_be_absent_from_input=True, must_be_absent_from_output=True,
    )
    concepts, _ = pp.resolve_concepts(tokenizer, prompt)
    violations = pp.protocol_violations(
        prompt, concepts,
        prompt_token_ids=tokenizer.encode("alpha beta"),
        generated_token_ids=tokenizer.encode("answer", add_special_tokens=False),
    )
    assert violations == ["intermediates present in input: ['alpha']"]


def test_item_level_pass_at_k_and_log_auc():
    ranks = [[0, 9], [4]]
    values = pp.item_pass_scores(ranks, [1, 5, 10])
    assert values == pytest.approx([0.25, 0.75, 1.0])
    assert 0 <= pp.log_k_auc([1, 5, 10], values) <= 1


def test_emergence_uses_band_synonyms_and_sustained_onset():
    # ranks: [position, layer, tracked-token-column], all zero-based.
    ranks = np.full((2, 5, 2), 100, dtype=int)
    ranks[0, :, 0] = [50, 8, 3, 2, 1]
    ranks[0, :, 1] = [40, 7, 6, 6, 6]
    slice_data = SimpleNamespace(
        tracked_token_ids=[11, 12],
        rank_tensor=ranks,
        layers=[0, 1, 2, 3, 4],
    )
    model = SimpleNamespace(n_layers=6)
    concept = pp.ConceptTokens("alpha", (11, 12), ("alpha", "beta"))
    result = matviz.concept_emergence(
        slice_data, model, [concept], positions=[0], band=(20, 80),
        threshold=5, sustain=2,
    )[0]
    assert result.best_rank == 1
    assert result.best_depth == 80
    assert result.onset_depth == 40


def test_lens_metadata_rejects_wrong_model():
    metadata = {"model": {"requested_id": "a", "model_class": "M",
                           "n_layers": 2, "d_model": 4, "vocab_size": 10}}
    with pytest.raises(ValueError, match="requested_id"):
        pp.validate_lens_metadata(
            metadata,
            identity={"requested_id": "b", "model_class": "M",
                      "n_layers": 2, "d_model": 4, "vocab_size": 10},
            require=True,
        )
