# Oriented cross-encoding: computation

This page defines the **oriented cross-encoding matrix** $M$ and how library code
aggregates pre-anchor encoder means into one square cell. It complements the
workflow guide [Oriented cross-encoding matrix](cross-encoding-matrix.md).

Paper appendix: [cross_encoding_matrix.tex](../paper/cross_encoding_matrix.tex).

---

## Setup

Let $\mathcal{F}:=\{f_1,\ldots,f_K\}$ be feature classes (e.g. `3SG`,
`Fem.Nom`) and $\mathcal{G}:=\{G_1,\ldots,G_M\}$ a set of pairwise
[GRADIEND](../api/training/Trainer.md) models (e.g. `3SG`↔`3PL`). Each
$G\in\mathcal{G}$ is trained on an ordered pair
$(a_G,b_G)\in\mathcal{F}^2$.

We evaluate each $G$ on a shared test pool $\mathcal{D}$. Each probe
$x\in\mathcal{D}$ is associated with a factual class
$\mathrm{fac}(x)\in\mathcal{F}$ and a counterfactual class
$\mathrm{cnf}(x)\in\mathcal{F}$. For $f,c\in\mathcal{F}$ with
$\mathcal{D}_{f,c}\neq\emptyset$, the directed transition slice is

$$
\mathcal{D}_{f,c}=\{x\in\mathcal{D}:\mathrm{fac}(x)=f,\ \mathrm{cnf}(x)=c\}.
$$

For model $G$, the **encoding mean** on that contrast slice is

$$
\bar{e}_G(f,c)=\frac{1}{|\mathcal{D}_{f,c}|}\sum_{x\in\mathcal{D}_{f,c}}\mathrm{enc}_G(x),
$$

where $\mathrm{enc}_G(x)$ is the scalar encoded feature value of probe $x$ under
$G$. In code, these are rows of `encoder_df` grouped by directed transition
`factual→counterfactual`; see
[`compute_gradiend_transition_cross_encoding_matrix`][gradiend.comparison.cross_encoding.compute_gradiend_transition_cross_encoding_matrix]
for the rectangular **pre-anchor** matrix (GRADIEND × transition).

For a probe feature $f_j$, let $\mu_G(f_j)$ be the mean encoded value of $G$ over
all probes whose GRADIEND source-side class is $f_j$:

$$
\mu_G(f_j)=
\begin{cases}
\operatorname{mean}\{\mathrm{enc}_G(x):\mathrm{fac}(x)=f_j\}, & \text{if }G\text{ uses factual gradients as input},\\
\operatorname{mean}\{\mathrm{enc}_G(x):\mathrm{cnf}(x)=f_j\}, & \text{if }G\text{ uses counterfactual gradients as input}.
\end{cases}
$$

---

## Rows vs columns

For each entry $M_{i,j}$ of the cross-encoding matrix $M$, $f_i$ is the
**orienting feature** and $f_j$ is the **probe feature**. We ask whether GRADIEND
models that separate $f_i$ from other features assign high encoder values to
$f_j$ probes.

| Role | Index | Meaning |
|------|-------|---------|
| **Row** $f_i$ | Orienting feature | Select GRADIENDs whose ordered pair contains $f_i$, then align their signs toward $f_i$ |
| **Column** $f_j$ | Probe feature | Select probes whose source-side class for the evaluated GRADIEND is $f_j$ |

Define $\mathcal{G}_{f_i}=\{G\in\mathcal{G}: f_i\in\{a_G,b_G\}\}$ and the
**anchor sign**

$$
\operatorname{sign}_G(f_i)=
\begin{cases}
+1 & \text{if } f_i=a_G \\
-1 & \text{if } f_i=b_G
\end{cases}
$$

The cross-encoding matrix is

$$
M_{i,j}
= \frac{1}{|\mathcal{G}_{f_i}|}
\sum_{G\in\mathcal{G}_{f_i}} \operatorname{sign}_G(f_i)\,\mu_G(f_j).
$$

Implementation:
[`compute_anchor_aligned_encoding_matrix`][gradiend.comparison.anchor_aligned.compute_anchor_aligned_encoding_matrix]
builds per-contribution rows in `aligned_rows`, then
[`aggregate_anchor_aligned_encoding_rows`][gradiend.comparison.anchor_aligned.aggregate_anchor_aligned_encoding_rows]
pivots to $M$.

---

## What $M$ is (and is not)

- $M$ is **not** a similarity matrix: $M_{f_i,f_j}\neq M_{f_j,f_i}$ in general
  because rows and columns aggregate different quantities. Rows aggregate models
  oriented toward a feature, whereas columns select the probes being encoded.
- **Diagonal** $M_{f_i,f_i}$: self-encoding strength when the orienting feature
  and probe feature match. It is not a perfect self-similarity score; diagonal
  entries are therefore typically below $1$.
- **Off-diagonal** $M_{f_i,f_j}$: cross-encoding from orienting feature $f_i$ to
  probe feature $f_j$ (leakage / shared structure). For inverted binary pairs,
  $M_{f_a,f_b}$ and $M_{f_b,f_a}$ typically have opposite signs.

In code, `alignment="auto"` follows the GRADIEND source side: factual-source
models use factual probe classes, and counterfactual-source models use
counterfactual probe classes. Explicit `alignment` values are diagnostic
overrides. See
[`encoding_view_sign_for_source`][gradiend.model._source_target.encoding_view_sign_for_source]
for the view sign applied before anchor aggregation.

---

## Null controls (planned)

Cross-encoding currently reports observed matrices and, for multi-seed runs,
seed-level dispersion. It does **not** yet test whether an observed association
is larger than expected when feature classes have no meaningful relationship.
Until null controls are implemented, off-diagonal patterns should therefore be
treated as hypotheses rather than confirmatory findings.

Planned null-control analyses, in implementation priority order, are:

1. **Label permutation:** shuffle probe feature labels while preserving class
   counts, recompute the matrix repeatedly, and compare each observed cell with
   its permutation distribution. This is the preferred first implementation.
2. **Sign/alignment permutation:** randomly flip pairwise GRADIEND orientations
   before anchor aggregation to test whether row structure depends on the
   intended sign frame.
3. **Random feature pairs:** construct size-matched artificial contrasts to
   measure cross-encoding structure for semantically meaningless splits.
4. **Matched lexical controls:** compare semantic or social features with
   random target-token sets matched for frequency, token length, and sample
   count.

This should be implemented as a first-class comparison analysis, not as a
heatmap-only option. A null-control result should retain the observed matrix and
report the null mean and standard deviation, empirical p-values or z-scores,
multiple-comparison correction, permutation count, random seed, and any
matching or stratification rules. No such API is implemented yet.

---

## API map

| Step | Function |
|------|----------|
| Shared test pool encoding | [`build_cross_task_encoder_summary`][gradiend.comparison.cross_encoding.build_cross_task_encoder_summary] |
| Pre-anchor GRADIEND × transition | [`compute_gradiend_transition_cross_encoding_matrix`][gradiend.comparison.cross_encoding.compute_gradiend_transition_cross_encoding_matrix] |
| Oriented rows + pivot | [`compute_anchor_aligned_encoding_matrix`][gradiend.comparison.anchor_aligned.compute_anchor_aligned_encoding_matrix] |
| Plot | [`plot_cross_encoding_heatmap`][gradiend.visualizer.heatmaps.encoding.plot_cross_encoding_heatmap] |
| Inspect contributions | `payload["aligned_rows"]` on oriented payload |

---

## Related

- [Oriented cross-encoding matrix](cross-encoding-matrix.md) — runnable pipeline and figures
- [Evaluation & visualization](evaluation-visualization.md) — heatmap customization
- [Cross-model comparison](cross-model-comparison.md) — when to use dense matrices
