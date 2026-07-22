# Frozen lexical-adversarial representation protocol

Frozen: `2026-07-17T11:46:24.759321+00:00`

## Scientific question

When wording and physics disagree, do Gemma's internal states become more similar for physically equivalent descriptions than for near-verbatim descriptions with the opposite physical relation?

## Design

The suite contains 24 independently parameterized materials triplets across six mechanism families. Each triplet contains (i) an anchor, (ii) a scientifically equivalent paraphrase with changed terminology and converted units, and (iii) a near-verbatim numerical reversal with the opposite physical answer. Answer order is fixed within a triplet and balanced across triplets.

Before freezing, both word and character TF-IDF selected the lexical counterfactual as the closer neighbor in all 24 triplets. This is a design preflight, not a model result.

## Frozen primary endpoint

At every registered layer, center the three-fit mean Jacobian target states across all 72 prompts and calculate cosine similarity. For each triplet, subtract anchor-to-counterfactual similarity from anchor-to-paraphrase similarity. Positive values mean that physical equivalence outranks lexical overlap. Average over the fixed 38--92% layer band and use the frozen two-stage family/triplet bootstrap.

## Secondary endpoints

- direct-unembedding-basis and raw-residual triplet margins
- paired Jacobian-minus-direct triplet margin
- fraction of triplets and family means with positive band margin
- clean answer accuracy and anchor-paraphrase-counterfactual consistency
- target-free top-30 word-set Jaccard margin at the five frozen layers after triplet-union prompt morphology filtering
- word and character TF-IDF lexical margins

## Guardrails

- Similarity is evidence of representation organization, not causal use.
- A positive result does not reveal a literal chain of thought.
- The suite tests six monotonic textbook relations, not unrestricted materials reasoning.
- No prompt, family, layer, or word is excluded after execution.

## Fingerprints

- prompt manifest: `bcb3fd2853de772a7d4a16c7f5535a7eaddcc91416dc7eca72e906229adf8197`
- execution runner: `de6659a0ca75af96bfaac955d8ed1c05c0f701ec99b0780087902a4de086f0b7`
- lens seed 0: `d15ff55233c458f4289a7aac1b3f5c8e6441d0334a44a7b6fce03e447889aa99`
- lens seed 1: `98bf7c7491c525df5ae9c9ac8040f450cce630dc8257a2ae062e6bdbf76980dd`
- lens seed 2: `51930e2b8d751de78e66ed92fcf6c1724783a4f81f94d0b7021d2278aabe00e5`
