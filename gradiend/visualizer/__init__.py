"""
Visualizer package: evaluation-related plots.

- Visualizer(trainer): holds trainer; exposes single-model plots.
- encoder_distributions: plot_encoder_distributions(trainer, encoder_df=None, ...)
- topk_venn: compute_topk_sets, plot_topk_venn, plot_topk_neuron_intersection
"""

__all__ = [
    "Visualizer",
    "plot_encoder_distributions",
    "plot_training_convergence",
    "plot_encoder_scatter",
    "plot_encoder_strip_by_split",
    "plot_encoder_by_target",
    "plot_encoder_component_artifacts",
    "plot_training_component_artifacts",
    "TokenEncodingRow",
    "compute_token_encodings",
    "highlight_token_encoding",
    "list_token_encoding_components",
    "render_token_encoding_html",
    "EncodingColorNorm",
    "resolve_encoding_color_norm",
    "check_plot_environment",
    "configure_plot_style",
    "PlotStyleConfig",
    "PlotStyleStatus",
    "format_transition_label",
    "transition_bidi_arrow",
    "transition_directed_arrow",
    "plot_comparison_heatmap",
    "plot_cross_encoding_heatmap",
    "plot_gradiend_feature_cross_encoding_heatmap",
    "plot_gradiend_transition_cross_encoding_heatmap",
    "plot_similarity_heatmap",
    "plot_similarity_heatmap_with_correlation",
    "compute_topk_sets",
    "plot_topk_venn",
    "plot_topk_overlap_heatmap",
    "plot_topk_overlap_venn",
]

_LAZY_IMPORTS = {
    "Visualizer": ("gradiend.visualizer.visualizer", "Visualizer"),
    "plot_encoder_distributions": ("gradiend.visualizer.encoder_distributions", "plot_encoder_distributions"),
    "plot_training_convergence": ("gradiend.visualizer.convergence", "plot_training_convergence"),
    "plot_encoder_scatter": ("gradiend.visualizer.encoder_scatter", "plot_encoder_scatter"),
    "plot_encoder_strip_by_split": ("gradiend.visualizer.encoder_strip_split", "plot_encoder_strip_by_split"),
    "plot_encoder_by_target": ("gradiend.visualizer.encoder_by_target", "plot_encoder_by_target"),
    "plot_encoder_component_artifacts": ("gradiend.visualizer.components", "plot_encoder_component_artifacts"),
    "plot_training_component_artifacts": ("gradiend.visualizer.components", "plot_training_component_artifacts"),
    "TokenEncodingRow": ("gradiend.visualizer.token_encoding", "TokenEncodingRow"),
    "compute_token_encodings": ("gradiend.visualizer.token_encoding", "compute_token_encodings"),
    "highlight_token_encoding": ("gradiend.visualizer.token_encoding", "highlight_token_encoding"),
    "list_token_encoding_components": ("gradiend.visualizer.token_encoding", "list_token_encoding_components"),
    "render_token_encoding_html": ("gradiend.visualizer.token_encoding", "render_token_encoding_html"),
    "EncodingColorNorm": ("gradiend.visualizer.color_norm", "EncodingColorNorm"),
    "resolve_encoding_color_norm": ("gradiend.visualizer.color_norm", "resolve_encoding_color_norm"),
    "check_plot_environment": ("gradiend.visualizer.plot_style", "check_plot_environment"),
    "configure_plot_style": ("gradiend.visualizer.plot_style", "configure_plot_style"),
    "PlotStyleConfig": ("gradiend.visualizer.plot_style_config", "PlotStyleConfig"),
    "PlotStyleStatus": ("gradiend.visualizer.plot_style_config", "PlotStyleStatus"),
    "format_transition_label": ("gradiend.visualizer.labels", "format_transition_label"),
    "transition_bidi_arrow": ("gradiend.visualizer.labels", "transition_bidi_arrow"),
    "transition_directed_arrow": ("gradiend.visualizer.labels", "transition_directed_arrow"),
    "plot_comparison_heatmap": ("gradiend.visualizer.heatmaps", "plot_comparison_heatmap"),
    "plot_cross_encoding_heatmap": ("gradiend.visualizer.heatmaps", "plot_cross_encoding_heatmap"),
    "plot_gradiend_feature_cross_encoding_heatmap": (
        "gradiend.visualizer.heatmaps",
        "plot_gradiend_feature_cross_encoding_heatmap",
    ),
    "plot_gradiend_transition_cross_encoding_heatmap": (
        "gradiend.visualizer.heatmaps",
        "plot_gradiend_transition_cross_encoding_heatmap",
    ),
    "plot_similarity_heatmap": ("gradiend.visualizer.heatmaps", "plot_similarity_heatmap"),
    "plot_similarity_heatmap_with_correlation": (
        "gradiend.visualizer.heatmaps",
        "plot_similarity_heatmap_with_correlation",
    ),
    "plot_topk_overlap_heatmap": ("gradiend.visualizer.topk", "plot_topk_overlap_heatmap"),
    "plot_topk_overlap_venn": ("gradiend.visualizer.topk", "plot_topk_overlap_venn"),
    "compute_topk_sets": ("gradiend.visualizer.topk.venn_", "compute_topk_sets"),
    "plot_topk_venn": ("gradiend.visualizer.topk.venn_", "plot_topk_venn"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        module_name, attr_name = _LAZY_IMPORTS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
