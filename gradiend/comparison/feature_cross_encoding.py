"""Compatibility alias for feature-wise cross-encoding helpers.

Feature cross-encoding was consolidated into :mod:`cross_encoding`.  This
module remains importable so older scripts and cached cluster jobs keep using
the same implementation (including monkeypatches of module globals).
"""

from __future__ import annotations

import sys

from gradiend.comparison import cross_encoding as _cross_encoding

sys.modules[__name__] = _cross_encoding
