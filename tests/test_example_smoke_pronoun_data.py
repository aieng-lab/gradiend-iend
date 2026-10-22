"""The example smoke test gives the slow examples smaller data (no network, no training)."""

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys

from gradiend.examples import create_english_pronoun_data as pronoun_data
from gradiend.trainer.core.arguments import TrainingArguments
from test_bench.examples import test_examples_smoke as smoke

_FAKE_PRONOUN_EXAMPLE = """import json
from gradiend.examples.create_english_pronoun_data import english_pronoun_generation_config

if __name__ == "__main__":
    print("CONFIG=" + json.dumps(english_pronoun_generation_config()))
"""

_FAKE_TRAINING_EXAMPLE = """import json
from gradiend import TrainingArguments

if __name__ == "__main__":
    own = TrainingArguments(experiment_dir="x", encoder_eval_train_max_size=7)
    plain = TrainingArguments(experiment_dir="x")
    print("ARGS=" + json.dumps({
        "own_encoder_cap": own.encoder_eval_train_max_size,
        "train_cap": plain.train_max_size,
        "encoder_cap": plain.encoder_eval_train_max_size,
        "neutral_cap": plain.decoder_eval_max_size_neutral,
    }))
"""


_FAKE_NEUTRAL_EXAMPLE = """import json
from gradiend.examples import english_pronoun_datasets as published

if __name__ == "__main__":
    loader = published.load_english_pronoun_neutral_data
    print("NEUTRAL=" + json.dumps({"wrapped": hasattr(loader, "__wrapped__"), "rows": getattr(loader, "neutral_rows", None)}))
"""


def _source(module_name):
    with open(importlib.util.find_spec(module_name).origin, encoding="utf-8") as handle:
        return handle.read()


def _run_runner(tmp_path, example_source, constants, caps, neutral_rows=0):
    (tmp_path / "fake_example.py").write_text(example_source)
    return subprocess.run(
        [sys.executable, "-c", smoke._SMOKE_RUNNER, "fake_example", json.dumps(constants), json.dumps(caps), str(neutral_rows)],
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(smoke.PROJECT_ROOT)])},
        capture_output=True,
        text=True,
        timeout=300,
    )


def _printed(result, prefix):
    line = next(line for line in result.stdout.splitlines() if line.startswith(prefix))
    return json.loads(line[len(prefix):])


def test_every_slow_example_is_covered():
    generates = {name for name in smoke.EXAMPLE_MODULES if "ensure_english_pronoun_data" in _source(name)}
    published = {name for name in smoke.EXAMPLE_MODULES if "english_pronoun_datasets" in _source(name)}
    assert generates == smoke.PRONOUN_DATA_EXAMPLES
    assert published == smoke.PUBLISHED_DATA_EXAMPLES
    assert smoke.PRONOUN_DATA_EXAMPLES.isdisjoint(smoke.PUBLISHED_DATA_EXAMPLES)


def test_settings_name_existing_fields():
    for name in smoke.SMALL_PRONOUN_DATA:
        assert hasattr(pronoun_data, name), name
    fields = {field.name for field in dataclasses.fields(TrainingArguments)}
    assert set(smoke.SMOKE_TRAINING_CAPS) <= fields
    # the final evaluations must keep their default size: it is part of the cache file name the examples read back
    assert "encoder_eval_max_size" not in smoke.SMOKE_TRAINING_CAPS


def test_small_corpus_is_small():
    settings = smoke.SMALL_PRONOUN_DATA
    full = pronoun_data.WIKIPEDIA_BASE_MAX_SIZE  # None: the whole of Wikipedia
    assert full is None or settings["WIKIPEDIA_BASE_MAX_SIZE"] < full
    assert settings["MAX_SIZE_PER_CLASS"] <= pronoun_data.MAX_SIZE_PER_CLASS
    assert settings["NEUTRAL_MAX_SIZE"] <= pronoun_data.NEUTRAL_MAX_SIZE


def test_other_examples_run_unchanged():
    assert smoke._example_command("gradiend.examples.start_workflow") == [
        sys.executable, "-m", "gradiend.examples.start_workflow",
    ]


def test_slow_examples_use_the_runner():
    for name in smoke.PRONOUN_DATA_EXAMPLES | smoke.PUBLISHED_DATA_EXAMPLES:
        command = smoke._example_command(name)
        assert command[:3] == [sys.executable, "-c", smoke._SMOKE_RUNNER]
        assert command[3] == name
    (generator,) = smoke.PRONOUN_DATA_EXAMPLES
    assert json.loads(smoke._example_command(generator)[4]) == smoke.SMALL_PRONOUN_DATA
    assert json.loads(smoke._example_command(generator)[5]) == {}
    assert smoke._example_command(generator)[6] == "0"
    trainer = sorted(smoke.PUBLISHED_DATA_EXAMPLES)[0]
    assert json.loads(smoke._example_command(trainer)[4]) == {}
    assert json.loads(smoke._example_command(trainer)[5]) == smoke.SMOKE_TRAINING_CAPS
    assert smoke._example_command(trainer)[6] == str(smoke.SMOKE_NEUTRAL_ROWS)


def test_runner_gives_the_example_the_small_corpus(tmp_path):
    result = _run_runner(tmp_path, _FAKE_PRONOUN_EXAMPLE, smoke.SMALL_PRONOUN_DATA, {})
    assert result.returncode == 0, result.stderr
    config = _printed(result, "CONFIG=")
    assert config["base_data"] == smoke.SMALL_PRONOUN_DATA["WIKIPEDIA_DATASET"]
    assert config["hf_config"] is None
    assert config["base_max_size"] == smoke.SMALL_PRONOUN_DATA["WIKIPEDIA_BASE_MAX_SIZE"]
    assert config["max_size_per_class"] == smoke.SMALL_PRONOUN_DATA["MAX_SIZE_PER_CLASS"]
    assert config["neutral_max_size"] == smoke.SMALL_PRONOUN_DATA["NEUTRAL_MAX_SIZE"]


def test_runner_caps_the_training_arguments_of_the_example(tmp_path):
    result = _run_runner(tmp_path, _FAKE_TRAINING_EXAMPLE, {}, smoke.SMOKE_TRAINING_CAPS)
    assert result.returncode == 0, result.stderr
    seen = _printed(result, "ARGS=")
    assert seen["train_cap"] == smoke.SMOKE_TRAINING_CAPS["train_max_size"]
    assert seen["encoder_cap"] == smoke.SMOKE_TRAINING_CAPS["encoder_eval_train_max_size"]
    assert seen["neutral_cap"] == smoke.SMOKE_TRAINING_CAPS["decoder_eval_max_size_neutral"]
    assert seen["own_encoder_cap"] == 7, "a value the example sets itself is kept"


def test_runner_cuts_the_neutral_texts_of_the_example(tmp_path):
    result = _run_runner(tmp_path, _FAKE_NEUTRAL_EXAMPLE, {}, {}, neutral_rows=17)
    assert result.returncode == 0, result.stderr
    assert _printed(result, "NEUTRAL=") == {"wrapped": True, "rows": 17}
    result = _run_runner(tmp_path, _FAKE_NEUTRAL_EXAMPLE, {}, {}, neutral_rows=0)
    assert _printed(result, "NEUTRAL=") == {"wrapped": False, "rows": None}


def test_runner_rejects_unknown_settings(tmp_path):
    result = _run_runner(tmp_path, _FAKE_PRONOUN_EXAMPLE, {"NO_SUCH_SETTING": 1}, {})
    assert result.returncode != 0
    assert "no setting 'NO_SUCH_SETTING'" in result.stderr

    result = _run_runner(tmp_path, _FAKE_TRAINING_EXAMPLE, {}, {"no_such_field": 1})
    assert result.returncode != 0
    assert "no field(s) ['no_such_field']" in result.stderr
