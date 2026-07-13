"""Save matplotlib figures for experiment scripts."""

from __future__ import annotations

import os
from typing import Any

from matplotlib import pyplot as plt

from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def save_figure(
    fig: plt.Figure,
    output_path: str,
    *,
    show: bool = False,
    **kwargs: Any,
) -> str:
    """Save a figure to disk; optionally call ``fig.show()`` for IDE/interactive viewers."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight", **kwargs}
    abs_path = os.path.abspath(output_path)
    try:
        fig.savefig(abs_path, **save_kwargs)
    except PermissionError:
        root, ext = os.path.splitext(abs_path)
        abs_path = f"{root}_new{ext}"
        fig.savefig(abs_path, **save_kwargs)
        logger.warning(
            "Could not overwrite %s (close viewer); wrote %s instead.",
            output_path,
            abs_path,
        )
    logger.info("Wrote %s", abs_path)
    if show:
        fig.show()
    else:
        plt.close(fig)
    return abs_path
