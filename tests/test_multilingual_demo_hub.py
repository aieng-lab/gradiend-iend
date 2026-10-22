"""A Hugging Face repo id as --experiment-dir is downloaded to runs/<name>; local folders stay as they are."""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def demo():
    import experiments.multilingual_gradiend_demo as mod

    return mod


def test_existing_local_folders_and_new_paths_are_kept(demo, tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr("huggingface_hub.HfApi.repo_exists", lambda self, *a, **k: called.append(a) or False)
    assert demo.resolve_experiment_dir(str(tmp_path)) == str(tmp_path)  # exists: no lookup
    assert demo.resolve_experiment_dir("runs/new_run") == "runs/new_run"  # no such repo
    assert demo.resolve_experiment_dir("single_name") == "single_name"
    assert called == [("runs/new_run",)]


def test_hub_repo_is_downloaded_into_runs(demo, monkeypatch):
    seen = {}
    monkeypatch.setattr("huggingface_hub.HfApi.repo_exists", lambda self, *a, **k: True)

    def fake_download(repo_id, **kwargs):
        seen.update(repo_id=repo_id, **kwargs)
        return kwargs["local_dir"]

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    out = demo.resolve_experiment_dir("aieng-lab/gradiend-multilingual-bert-features")
    assert seen["repo_id"] == "aieng-lab/gradiend-multilingual-bert-features"
    assert out.replace("\\", "/") == "runs/gradiend-multilingual-bert-features"


def test_offline_lookup_falls_back_to_a_local_folder(demo, monkeypatch):
    def offline(self, *a, **k):
        raise OSError("no network")

    monkeypatch.setattr("huggingface_hub.HfApi.repo_exists", offline)
    assert demo.resolve_experiment_dir("aieng-lab/some-repo") == "aieng-lab/some-repo"


def test_hub_download_is_detected(demo, tmp_path):
    assert demo._is_hub_download("aieng-lab/gradiend-x", "runs/gradiend-x")
    assert not demo._is_hub_download("runs/new_run", "runs/new_run")  # nothing was downloaded
    assert not demo._is_hub_download(str(tmp_path), "elsewhere")  # an existing local folder
    assert not demo._is_hub_download(None, "runs/x")
