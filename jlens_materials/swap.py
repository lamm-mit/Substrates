# Copyright 2026.  Apache-2.0.
"""Causal J-lens coordinate swaps (paper Figure 4C / verbal-report swap).

The read-out figures show what an activation is *disposed* to say. This module
does the causal half: it edits the residual stream along J-lens directions and
checks whether the model's *output* changes accordingly.

J-lens direction for token ``t`` at layer ``l``
-----------------------------------------------
A layer-``l`` residual ``h`` produces the pre-softmax lens logit for token ``t``
as ``<W_U[t], J_l @ h> = <J_l^T W_U[t], h>``.  So the residual-space direction
that carries token ``t``'s J-lens coordinate is::

    v_{l,t} = normalize(J_l^T @ W_U[t])

(the final-norm Jacobian is folded in only approximately — the same
approximation the paper's steering uses: "the unit-normalized transpose row for
that token").

Two interventions
-----------------
* ``projection`` : the original approximation in this repo,
  ``h += <h, v_s>(alpha * v_t - v_s)``.  This is exact only when the source and
  target vectors are effectively orthogonal.
* ``pinv``       : the paper-style coordinate swap.  Form
  ``V = [v_s, v_t]``, solve ``c = V† h`` for the source/target coordinates, set
  ``c' = [c_t, alpha * c_s]``, and patch ``h' = h + V(c' - c)``.  The component
  of ``h`` orthogonal to ``span(v_s, v_t)`` is unchanged.

``run_swap`` reports the model's next-token distribution at the final position
with and without the swap. Rank movement is retained as a diagnostic, but a
paper-style causal success requires a correct clean answer and the declared
counterfactual answer after intervention.

    python swap.py --model Qwen/Qwen2.5-0.5B-Instruct --lens lens_qwen2forcausallm.pt \
        --prompt "Think of a metal failure mode. Answer in one word." \
        --source fatigue --target creep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# use the vendored jlens (committed under _vendor_jlens/) without installing
_REPO = Path(__file__).resolve().parent / "_vendor_jlens"
if (_REPO / "jlens").is_dir():
    sys.path.insert(0, str(_REPO))
import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402


def band_layers(lens, n_layers: int, lo: float = 0.38, hi: float = 0.92) -> list[int]:
    """Fitted layers in a preregistered band (fractions or depth percentages)."""
    if lo <= 1 and hi <= 1:
        lo, hi = lo * 100.0, hi * 100.0
    if not 0 <= lo <= hi <= 100:
        raise ValueError(f"invalid workspace band {(lo, hi)}")
    denom = max(n_layers - 1, 1)
    return [layer for layer in lens.source_layers
            if lo <= 100.0 * layer / denom <= hi]


def jlens_direction(lens, model, layer: int, token_id: int) -> torch.Tensor:
    """Unit residual-space direction carrying token ``token_id``'s J-lens
    coordinate at ``layer``: ``normalize(J_l^T @ W_U[token_id])``."""
    W_U = model._lm_head.weight  # [vocab, d_model]
    w_t = W_U[token_id].float().to("cpu")           # [d_model]
    J = lens.jacobians[layer].float()               # [d_model, d_model]
    v = J.T @ w_t                                    # J_l^T W_U[t]
    return (v / (v.norm() + 1e-8))


def _single_token(tokenizer, word: str) -> int:
    for f in (" " + word, word, " " + word.capitalize(), word.capitalize()):
        ids = tokenizer.encode(f, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"{word!r} is not a single token for this model (paper §9.1)")


def _answer_token_ids(tokenizer, answer: str | None) -> set[int]:
    if not answer:
        return set()
    ids = set()
    for form in (" " + answer, answer, " " + answer.capitalize(), answer.capitalize(),
                 " " + answer.lower(), answer.lower()):
        encoded = tokenizer.encode(form, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    return ids


class SwapHook:
    """Forward hooks on band layers that apply a J-lens coordinate swap to the
    residual at the given positions on the next forward pass."""

    def __init__(self, model, lens, layers, positions, v_by_layer, alpha,
                 method: str):
        self._blocks = model.layers
        self._layers = layers
        self._positions = positions
        self._v = v_by_layer  # {layer: (v_source, v_target)} on the block's device
        self._alpha = alpha
        self._method = method
        self._handles = []

    def _make(self, layer):
        v_s, v_t, V_cpu, V_pinv_cpu = self._v[layer]

        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            dev, dt = tensor.device, tensor.dtype
            h = tensor[:, self._positions, :]                     # [b, P, d]
            if self._method == "projection":
                vs = v_s.to(dev, dt)
                vt = v_t.to(dev, dt)
                coord = (h * vs).sum(-1, keepdim=True)            # <h, v_s>
                delta = coord * (self._alpha * vt - vs)           # swap s -> t
            else:
                V = V_cpu.to(dev, torch.float32)                  # [d,2]
                V_pinv = V_pinv_cpu.to(dev, torch.float32)        # [2,d]
                h_f = h.float()
                coords = h_f @ V_pinv.T                           # [b,P,2]
                swapped = torch.stack(
                    (coords[..., 1], self._alpha * coords[..., 0]), dim=-1
                )
                delta = (swapped - coords) @ V.T                  # [b,P,d]
                delta = delta.to(dt)
            tensor[:, self._positions, :] = h + delta
            return output

        return hook

    def __enter__(self):
        for layer in self._layers:
            self._handles.append(
                self._blocks[layer].register_forward_hook(self._make(layer)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def run_swap(
    model,
    lens,
    prompt: str,
    source: str,
    target: str,
    *,
    alpha: float = 1.0,
    top: int = 8,
    band=(0.38, 0.92),
    method: str = "pinv",
    positions: list[int] | None = None,
    expected_answer: str | None = None,
    acceptable_answers: tuple[str, ...] = (),
    swap_answer: str | None = None,
    source_token_id: int | None = None,
) -> dict:
    """Apply a coordinate swap and grade its declared causal outcome.

    Omitting ``source``/``source_token_id`` implements the verbal-report
    protocol: the model's clean greedy next token becomes the swap source.
    Probe-swap items should provide both ``expected_answer`` and
    ``swap_answer``; ``causal_success`` is true only when both outcomes match.
    """
    if method not in {"pinv", "projection"}:
        raise ValueError("method must be 'pinv' or 'projection'")
    tok = model.tokenizer
    t_id = _single_token(tok, target)
    layers = band_layers(lens, model.n_layers, *band)
    if not layers:
        raise ValueError("no fitted layers in the requested band")

    input_ids = model.encode(prompt, max_length=256)
    seq_len = input_ids.shape[1]
    positions = list(range(seq_len)) if positions is None else list(positions)
    if not positions or any(pos < 0 or pos >= seq_len for pos in positions):
        raise ValueError(f"swap positions must lie in [0, {seq_len}); got {positions}")

    def final_probs() -> torch.Tensor:
        with ActivationRecorder(model.layers, at=[model.n_layers - 1]) as rec:
            model.forward(input_ids)
            resid = rec.activations[model.n_layers - 1][0, -1:].float()
        return model.unembed(resid).float()[0].softmax(-1)

    clean = final_probs()
    dynamic_source = source_token_id is None and not source
    if source_token_id is None and source:
        s_id = _single_token(tok, source)
    elif source_token_id is not None:
        s_id = int(source_token_id)
    else:
        s_id = int(clean.argmax())
        source = tok.decode([s_id], clean_up_tokenization_spaces=False).strip()
    if s_id == t_id:
        raise ValueError("source and target resolve to the same token")

    def mats_for(layer):
        v_s = jlens_direction(lens, model, layer, s_id)
        v_t = jlens_direction(lens, model, layer, t_id)
        V = torch.stack([v_s, v_t], dim=1).float()
        return v_s, v_t, V, torch.linalg.pinv(V)

    v_by_layer = {layer: mats_for(layer) for layer in layers}

    def topk(probs):
        idx = probs.topk(top).indices
        return [(tok.decode([int(i)]).strip(), float(probs[int(i)])) for i in idx]

    def rank_of(probs, token_id):
        return int((probs > probs[token_id]).sum())

    with SwapHook(model, lens, layers, positions, v_by_layer, alpha, method):
        swapped = final_probs()

    clean_answer_ids = set()
    for answer in (expected_answer, *acceptable_answers):
        clean_answer_ids.update(_answer_token_ids(tok, answer))
    swap_answer_ids = _answer_token_ids(tok, swap_answer)
    clean_top_id, swapped_top_id = int(clean.argmax()), int(swapped.argmax())
    baseline_correct = (clean_top_id in clean_answer_ids
                        if clean_answer_ids else None)
    counterfactual_correct = (swapped_top_id in swap_answer_ids
                              if swap_answer_ids else None)
    causal_success = (
        bool(baseline_correct and counterfactual_correct)
        if baseline_correct is not None and counterfactual_correct is not None
        else None
    )
    target_top1 = swapped_top_id == t_id
    protocol_success = causal_success if causal_success is not None else (
        target_top1 if dynamic_source else None
    )

    return {
        "prompt": prompt,
        "source": source,
        "target": target,
        "alpha": alpha,
        "method": method,
        "band_layers": layers,
        "swap_positions": positions,
        "clean_top": topk(clean),
        "swapped_top": topk(swapped),
        "source_rank_clean": rank_of(clean, s_id),
        "source_rank_swapped": rank_of(swapped, s_id),
        "target_rank_clean": rank_of(clean, t_id),
        "target_rank_swapped": rank_of(swapped, t_id),
        "expected_answer": expected_answer,
        "swap_answer": swap_answer,
        "baseline_correct": baseline_correct,
        "counterfactual_correct": counterfactual_correct,
        "causal_success": causal_success,
        "target_top1": target_top1,
        "protocol_success": protocol_success,
        "dynamic_source": dynamic_source,
    }


_DTYPES = {"float32": torch.float32,
           "bfloat16": torch.bfloat16,
           "float16": torch.float16}


def _load(model_name, lens_path, dtype=torch.float32):
    import transformers
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype).to(dev)
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(lens_path)
    return model, lens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--source", default=None,
                    help="swap-out token; omit to use the clean greedy answer")
    ap.add_argument("--target", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--answer", default=None, help="expected clean answer")
    ap.add_argument("--swap-answer", default=None,
                    help="expected counterfactual answer after the swap")
    ap.add_argument("--workspace-band", default="38,92", metavar="LO,HI")
    ap.add_argument("--method", choices=["pinv", "projection"], default="pinv")
    ap.add_argument("--dtype", choices=list(_DTYPES), default="float32",
                    help="model weight dtype (default float32; bfloat16 ~halves "
                         "memory on GPU). The lens .pt is loaded in float32 "
                         "regardless.")
    args = ap.parse_args()
    model, lens = _load(args.model, args.lens, dtype=_DTYPES[args.dtype])
    band = tuple(float(x) for x in args.workspace_band.split(","))
    if len(band) != 2:
        raise SystemExit("--workspace-band must be LO,HI")
    out = run_swap(model, lens, args.prompt, args.source or "", args.target,
                   alpha=args.alpha, method=args.method, band=band,
                   expected_answer=args.answer, swap_answer=args.swap_answer)
    print(f"\nPrompt: {out['prompt']!r}")
    print(f"Swap J-lens coordinate: {out['source']} -> {out['target']} "
          f"(alpha={out['alpha']}, method={out['method']}) across band layers "
          f"{out['band_layers']}\n")
    print(f"{'CLEAN next-token':<34}{'AFTER SWAP'}")
    for (cw, cp), (sw, sp) in zip(out["clean_top"], out["swapped_top"]):
        print(f"  {cw:<20}{cp:>6.3f}      {sw:<20}{sp:>6.3f}")
    print(f"\n  source '{out['source']}': rank {out['source_rank_clean']+1} "
          f"-> {out['source_rank_swapped']+1}  (higher = more suppressed)")
    print(f"  target '{out['target']}': rank {out['target_rank_clean']+1} "
          f"-> {out['target_rank_swapped']+1}  (lower = more installed)")
    if out["causal_success"] is not None:
        print(f"  clean answer correct: {out['baseline_correct']}; "
              f"counterfactual correct: {out['counterfactual_correct']}; "
              f"causal success: {out['causal_success']}")


if __name__ == "__main__":
    main()
