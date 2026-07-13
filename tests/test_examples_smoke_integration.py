"""Expose the existing example smoke runner through the main pytest suite."""

import subprocess
import sys

import pytest

from test_bench.examples import test_examples_smoke as example_smoke


_CUDA_PREFLIGHT: subprocess.CompletedProcess[str] | None = None


def _require_cuda_in_fresh_process(module_name: str) -> None:
    """Check CUDA once in the same kind of fresh process used by the examples."""
    global _CUDA_PREFLIGHT
    if _CUDA_PREFLIGHT is None:
        _CUDA_PREFLIGHT = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os, torch; "
                    "print(f'CUDA_VISIBLE_DEVICES={os.environ.get(\"CUDA_VISIBLE_DEVICES\")!r}'); "
                    "print(f'torch.version.cuda={torch.version.cuda!r}'); "
                    "print(f'torch.cuda.device_count()={torch.cuda.device_count()}'); "
                    "assert torch.cuda.is_available(), 'CUDA is unavailable in fresh example process'; "
                    "print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

    if _CUDA_PREFLIGHT.returncode == 0:
        return

    details = (_CUDA_PREFLIGHT.stdout + _CUDA_PREFLIGHT.stderr).strip()
    if module_name == example_smoke.EXAMPLE_MODULES[0]:
        pytest.fail(f"CUDA preflight failed before first example:\n{details}")
    pytest.skip("CUDA preflight failed before the first example")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("module_name", example_smoke.EXAMPLE_MODULES)
def test_example_runs_successfully(module_name: str) -> None:
    """Run the test-bench example check when ``pytest tests/`` selects slow tests."""
    _require_cuda_in_fresh_process(module_name)
    example_smoke.test_example_runs_successfully(module_name)
