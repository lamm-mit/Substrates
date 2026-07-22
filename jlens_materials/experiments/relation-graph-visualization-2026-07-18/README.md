# Relation-graph publication visualizations

This is a plotting-only stage for the frozen natural-question, positional, and
cross-mechanism graph audits. It performs no model inference and does not select
an endpoint.

After regenerating the upstream state arrays and analysis tables, run from
`jlens_materials`:

```bash
python scripts/plot_relation_graph_robustness.py
```

The script creates the four-panel robustness figure, the 25-layer graph atlas,
the edge-persistence graph, and their machine-readable edge tables. Those
derived outputs are intentionally absent from this source-only release.

Nodes are exact prompts. Node shape denotes surface variant; fill denotes the
registered positive/negative outcome orientation. Teal edges preserve the
physically correct outcome and coral edges do not. The six islands arise
because the frozen candidate rule supplies the mechanism family; they are not
discovered communities.
