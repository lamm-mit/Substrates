# Documented exploratory protocol for paired held-out semantic streams

Status: retrospective visualization designed after the held-out vocabulary
results were inspected. It is not a preregistered or population-level endpoint.
The complete ten-family candidate grid is the population display; these paired
streams are case studies that explain what the filters do across layer depth.

## Scientific question

Do visually rich unrestricted streams mainly reflect generic connective words,
and which materials-relevant words remain after input leakage, common scaffold
tokens, and fit-specific variation are removed?

## Immutable inputs

- The three raw held-out run files in `runs/gemma4-e4b-it-heldout-v1-seed*.json`.
- The fixed final-prompt score position and 25 stored layer readouts.
- The frozen 214-word target-agnostic function list from
  `scripts/analyze_materials_heldout_v1.py`.
- The global scaffold learned by the already completed target-free population
  analysis.
- No model rerun and no lookup of the predeclared concept list.

## Left column: unrestricted single-fit rendering

1. Use lens fit 0 only.
2. Lowercase and retain ASCII alphabetic word-start tokens of length at least
   three; merge capitalization duplicates within a layer.
3. Apply no input-word, output-word, function-word, or scaffold filter.
4. For a stored list of width `W`, assign a token at rank index `r` the score
   `(W-r)/W` at that layer.
5. Select the seven tokens with the largest depth-integrated score over all
   registered layers, then order them by first appearance for the stream.

This column intentionally recreates the behavior of the original rich stream:
generic discourse words are allowed to dominate.

## Right column: strict three-fit-consensus rendering

1. Normalize tokens as above.
2. Remove tokens occurring in the corresponding prompt or one-token
   continuation.
3. Remove the frozen 214-word function list and the global target-free
   scaffold.
4. At each layer, retain only the intersection of the remaining lists from all
   three independently fitted lenses.
5. Score a surviving token by its mean reciprocal within-list rank across the
   three fits.
6. Select the seven tokens with the largest integrated score inside the fixed
   38--92% analysis band, then order them by first appearance.

No scientific word is added to the filter or supplied to either selector.
Residual connective words and tokenizer fragments remain visible rather than
being manually removed after inspection.

## Five displayed prompts

- `heldout-v1-assoc-boundary-attack-05`: selected as the strongest controlled
  positive family and because the strict stream contains `corrosion`.
- `heldout-v1-assoc-notch-resistance-01`: selected to show the emergent property
  neighborhood `resilience` / `protection`.
- `heldout-v1-assoc-line-defect-motion-04`: selected to show the mechanistic
  neighborhood `irreversible` / `mechanism`.
- `heldout-v1-assoc-ductile-03`: selected to show the failure-process
  neighborhood `collapse` / `disintegration`.
- `heldout-v1-assoc-cleavage-03`: selected as the shared negative family in the
  blinded assessment; its strict stream remains generic.

The choices are deliberately described as post hoc illustrative cases and do
not estimate how often such neighborhoods occur.

## Audit trail

`experiments/materials-heldout-v1_semantic_streams.json` stores the exact five
prompts, raw-run SHA-256 hashes, both protocols, every selected token and score,
and per-layer strict-consensus counts. The plotting implementation is
`scripts/build_final_paper_figures.py`.

## Interpretation boundary

Ribbon thickness is a rank-derived display score. A gap means that none of the
seven displayed words survived the strict rule at that layer; it does not mean
the model was inactive. The streams are not probabilities, attention maps,
causal pathways, private prose, or a literal chain of thought.
