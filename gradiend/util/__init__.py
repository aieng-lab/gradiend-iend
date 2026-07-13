"""Utility helpers exposed lazily to keep lightweight imports cheap."""

__all__ = [
    "get_logger",
    "setup_logging",
    "convert_tuple_keys_recursively",
    "restore_tuple_keys_recursively",
    "normalize_split_name",
    "json_loads",
    "to_jsonable",
    "hash_model_weights",
    "hash_it",
    "unwrap_model",
    "format_count",
    "cuda_unusable_runtime_error",
    "validate_cuda_usable_if_visible",
]

_LAZY_IMPORTS = {
    "get_logger": ("gradiend.util.logging", "get_logger"),
    "setup_logging": ("gradiend.util.logging", "setup_logging"),
    "convert_tuple_keys_recursively": ("gradiend.util.util", "convert_tuple_keys_recursively"),
    "restore_tuple_keys_recursively": ("gradiend.util.util", "restore_tuple_keys_recursively"),
    "normalize_split_name": ("gradiend.util.util", "normalize_split_name"),
    "json_loads": ("gradiend.util.util", "json_loads"),
    "to_jsonable": ("gradiend.util.util", "to_jsonable"),
    "hash_model_weights": ("gradiend.util.util", "hash_model_weights"),
    "hash_it": ("gradiend.util.util", "hash_it"),
    "unwrap_model": ("gradiend.util.util", "unwrap_model"),
    "format_count": ("gradiend.util.util", "format_count"),
    "cuda_unusable_runtime_error": ("gradiend.util.device", "cuda_unusable_runtime_error"),
    "validate_cuda_usable_if_visible": ("gradiend.util.device", "validate_cuda_usable_if_visible"),
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
