import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ablation():
    from experiments import gender_de_pruning_parameter_ablation as mod

    return mod


def test_default_mode_is_mask_recall(ablation):
    args = ablation.build_arg_parser().parse_args([])
    assert args.recall_only is True
    assert args.output_dir == ablation.DEFAULT_OUTPUT_DIR
    assert args.output_dir.endswith("german_de_v3")


def test_recall_only_flag_accepted(ablation):
    args = ablation.build_arg_parser().parse_args(["--recall-only"])
    assert args.recall_only is True


def test_pre_sources_default_order(ablation):
    assert ablation.PRE_SOURCES == ["factual", "alternative", "diff"]


def test_full_grid_disables_mask_recall(ablation):
    args = ablation.build_arg_parser().parse_args(["--full-grid"])
    assert args.recall_only is False
