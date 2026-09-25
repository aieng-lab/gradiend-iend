from gradiend.model import GradiendModel
from gradiend.trainer.core.callbacks import NormalizationCallback
from gradiend.trainer.core.signals import Signal
from gradiend.trainer.core.training import format_training_start_message


class _Model:
    def __init__(self, gradiend):
        self.gradiend = gradiend

    def __len__(self):
        return 123


def test_training_start_message_describes_component_split_activation_space():
    components = [
        {
            "id": f"activation:transformer.h.{index - 1}" if index else "activation:transformer.wte",
            "start": index * 768,
            "end": (index + 1) * 768,
        }
        for index in range(13)
    ]
    gradiend = GradiendModel(
        input_dim=9984,
        latent_dim=1,
        component_slices=components,
        component_split_mode="tensors",
        mapping_kind="activation",
    )
    args = type("Args", (), {"signal": Signal.activation(), "gradiend_split_loss": "mean"})()

    message = format_training_start_message(_Model(gradiend), args)

    assert message == (
        "Training component-split ACTIEND over 9,984 activation dimensions "
        "across 13 components with 1 feature neuron per component "
        "(loss aggregation=mean) "
        "[first 3: activation:transformer.wte, activation:transformer.h.0, activation:transformer.h.1]."
    )


def test_training_start_message_describes_non_split_gradient_entries():
    gradiend = GradiendModel(
        input_dim=123,
        latent_dim=2,
    )
    args = type("Args", (), {"signal": Signal.gradient()})()

    message = format_training_start_message(_Model(gradiend), args)

    assert message == "Training GRADIEND over 123 gradient entries with 2 feature neurons."


def test_component_normalization_log_uses_model_method_name(caplog):
    class _ComponentView:
        def invert_encoding(self):
            pass

    class _Gradiend:
        latent_dim = 1
        has_component_split = True
        method_name = "ACTIEND"

        def _component_view(self, _index):
            return _ComponentView()

    model = _Model(_Gradiend())
    callback = NormalizationCallback()
    caplog.set_level("INFO")
    eval_result = {
        "components": {
            "summary": {"n_components": 1},
            "metrics_by_component": {
                "activation:transformer.h.2": {
                    "component_index": 0,
                    "component_label": "transformer.h.2",
                    "correlation": -0.7,
                    "mean_by_class": {"-1.0": 0.8, "1.0": -0.8},
                }
            },
        }
    }

    callback.on_step_end(
        step=100,
        loss=1.0,
        model=model,
        config={
            "convergent_score_threshold": 0.5,
            "convergent_mean_by_class_threshold": 0.5,
        },
        eval_result=eval_result,
        training_stats={
            "components": {100: eval_result["components"]},
            "component_summary": {100: {}},
        },
    )

    assert "Inverted 1 ACTIEND component encoding: activation:transformer.h.2" in caplog.text
