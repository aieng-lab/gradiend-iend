# Comparison

Matrix computation helpers used with visualization and cross-model analysis:

- **[`compute_similarity_matrix`][gradiend.comparison.similarity.compute_similarity_matrix]** — Pairwise similarity between models
- **[`compute_grouped_similarity_matrices`][gradiend.comparison.similarity.compute_grouped_similarity_matrices]** — Grouped similarity matrices (e.g. by layer or component)
- **[`compute_gradiend_feature_cross_encoding_matrix`][gradiend.comparison.cross_encoding.compute_gradiend_feature_cross_encoding_matrix]** — GRADIEND × feature-class cross-encoding matrix
- **[`compute_gradiend_transition_cross_encoding_matrix`][gradiend.comparison.cross_encoding.compute_gradiend_transition_cross_encoding_matrix]** — GRADIEND × transition matrix (pre-anchor)
- **[`compute_anchor_aligned_encoding_matrix`][gradiend.comparison.anchor_aligned.compute_anchor_aligned_encoding_matrix]** — Oriented feature × feature matrix
- **[`compute_trainer_pair_encoding_matrix`][gradiend.comparison.trainer_pair_encoding.compute_trainer_pair_encoding_matrix]** — Trainer × trainer matrix for binary positive-pair suites

See also the [cross-model comparison guide](../../guides/cross-model-comparison.md) and [oriented cross-encoding matrix guide](../../guides/cross-encoding-matrix.md).
