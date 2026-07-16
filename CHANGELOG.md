# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.1] - 2026-07-16

### Changed

- Multi-seed dispersion heatmaps now use dispersion-specific autoscaling and one-decimal percentage annotations, making small seed-to-seed variation readable.
- Documentation now explains dispersion heatmap values as standard deviations of percentage overlap across seeds.
- Decoder evaluation cache keys now include split and size caps so cached grids are reused only for matching evaluation settings.

### Fixed

- `evaluate_decoder()` consistently accepts `split` and shared `max_size` arguments; `max_size` caps both training-like and neutral decoder evaluation rows unless explicit caps are supplied.
- Text-prediction and classification decoder evaluation now honor requested splits.
- Decoder plotting analysis honors `TrainingArguments.decoder_eval_max_size_training_like`, `decoder_eval_max_size_neutral`, and `eval_batch_size`, preventing uncapped neutral/LMS evaluation during probability-shift plotting.
- Encoder metrics caches are reused when `Trainer.evaluate_encoder()` supplies a trainer-owned cached encoder DataFrame.
- Multi-seed per-call `return_per_seed=True` is handled by `MultiSeedTrainerView` instead of leaking into single-seed trainer calls.
- Multi-seed encoder distribution, scatter, and strip-by-split plots now build per-seed encoder DataFrames before plotting.
- API autoref links in docs were refreshed for CI.

## [0.2.0] - 2026-07-13

### Added

- **Trainer suites** — [`TrainerSuite`][gradiend.trainer.suite.base.TrainerSuite], [`PositiveTrainerSuite`][gradiend.trainer.suite.positive.PositiveTrainerSuite], [`SymmetricTrainerSuite`][gradiend.trainer.suite.symmetric.SymmetricTrainerSuite], and [`TrainerCollection`][gradiend.trainer.suite.collection.TrainerCollection] for training and comparing multiple GRADIEND models from declarative pair/feature definitions.
- **Multi-seed analysis** — [`Trainer.multi_seed()`][gradiend.trainer.trainer.Trainer.multi_seed] and [`MultiSeedTrainerView`][gradiend.trainer.core.multi_seed.MultiSeedTrainerView] for seed aggregation, selection, and comparison plots.
- **Cross-model comparison** — similarity, cross-encoding, anchor-aligned, and GRADIEND-feature cross-encoding matrix helpers exported from `gradiend` (`compute_similarity_matrix`, `compute_grouped_similarity_matrices`, `compute_cross_encoding_matrix`, `compute_anchor_aligned_encoding_matrix`, `compute_gradiend_feature_cross_encoding_matrix`, `compute_gradiend_transition_cross_encoding_matrix`).
- **Plot styling API** — [`PlotStyleConfig`][gradiend.visualizer.plot_style_config.PlotStyleConfig], [`configure_plot_style()`][gradiend.visualizer.plot_style.configure_plot_style], transition label helpers ([`format_transition_label`][gradiend.visualizer.labels.format_transition_label]), `GRADIEND_PLOT_*` environment variables, and automatic `amsmath`/`amssymb` preamble when LaTeX is enabled. Guide: [Plot styling & LaTeX](guides/plot-styling-latex.md).
- **Comparison visualizations** — heatmap helpers (`plot_similarity_heatmap`, `plot_cross_encoding_heatmap`, `plot_gradiend_feature_cross_encoding_heatmap`, `plot_gradiend_transition_cross_encoding_heatmap`, `plot_comparison_heatmap`, `plot_topk_overlap_heatmap`, `plot_topk_overlap_venn`) and [`check_plot_environment`][gradiend.visualizer.plot_style.check_plot_environment].
- **Training cache policy** — `use_cache="only_convergent"` and training checkpoint fingerprinting (`cache_fingerprint` in `training.json`) for safer checkpoint reuse.
- **Data splits** — unified split columns, vocabulary-held-out splits, split-policy validation, and balancing helpers.
- **Transition selection** — [`TransitionSpec`][gradiend.trainer.core.transition_selection.TransitionSpec], [`pair()`][gradiend.trainer.core.transition_selection.pair], and [`identity()`][gradiend.trainer.core.transition_selection.identity] for explicit encoder-eval transition lists.
- **Decoder evaluation** — row-wise targets, explicit target validation, and richer probability-shift plotting.
- **Decoder-only MLM head** — auxiliary head training/saving and pooling-length ablation support.
- **Seq2seq objectives** — `seq2seq_decoder`, `seq2seq_encoder_mlm`, and `seq2seq_decoder_sequence_cloze` prediction objectives (experimental).
- **Runtime monitor** — optional heartbeat and CUDA OOM logging during training.
- **Documentation** — guides for trainer suites, multi-seed analysis, cross-model comparison, oriented cross-encoding matrices, data splits, decoder eval targets, token prediction methods, and plot styling/LaTeX.

### Changed

- Top-level `gradiend` exports expanded (suites, comparison, visualization helpers, plot styling); optional-import failures surface a clearer lazy error via `__getattr__`.
- Default `TextPredictionDataCreator.min_left_context_words` is now `10` (use `0` when sentence-initial tokens must match).
- Split-policy validation runs at training preparation time rather than on raw data load.
- `TextClassificationDataCreator` removed from top-level exports (still available under `gradiend.data.text.classification`).

### Deprecated

- Heatmap **`fmt`** argument — use **`annot_fmt`** instead (`plot_comparison_heatmap`, `plot_similarity_heatmap`, `plot_topk_overlap_heatmap`, and related wrappers).
- Method argument **`use_all_transitions`** — use **`include_other_classes`** instead (same behavior: include all class transitions in encoder evaluation when `len(all_classes) > 2`). Configure the default via [`TrainingArguments.include_other_classes`][gradiend.trainer.core.arguments.TrainingArguments].
- **`configure_matplotlib_style()`** — use **`configure_plot_style()`** instead.

### Fixed

- Trainer suite construction with explicit pair definitions no longer loads unresolved Hugging Face dataset ids during validation.
- Decoder probability selection recognizes unified `alternative_class` as well as legacy `alternative_id`.
- `ModelWithGradiend` deepcopy/pickle restores gradient locks correctly.
- Python 3.9 compatibility across trainer, comparison, and visualization code paths.
- CI unit tests no longer import from gitignored `experiments` trees at module level.
- LaTeX transition arrows (`\rightleftarrows`) include `amsmath` and `amssymb` in the matplotlib preamble; checkpoint `model.source` is no longer overwritten from `TrainingArguments` when loading finished checkpoints.

## [0.1.0] - 2025-02-04

Initial public release of the GRADIEND Python package.

[Unreleased]: https://github.com/aieng-lab/gradiend/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/aieng-lab/gradiend/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/aieng-lab/gradiend/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/aieng-lab/gradiend/releases/tag/v0.1.0
