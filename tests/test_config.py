import pytest
from pathlib import Path
from fairlens.config import load_config, EvalConfig, ModelConfig


def test_load_config(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(
        "probe_dir: my_probes\n"
        "output_dir: my_output\n"
        "num_generations: 3\n"
        "seed: 123\n"
        "models:\n"
        "  - name: gpt2\n"
        "    hf_id: gpt2\n"
        "    max_new_tokens: 20\n"
    )
    config = load_config(cfg_file)
    assert config.probe_dir == "my_probes"
    assert config.output_dir == "my_output"
    assert config.num_generations == 3
    assert config.seed == 123
    assert len(config.models) == 1
    assert config.models[0].hf_id == "gpt2"
    assert config.models[0].max_new_tokens == 20


def test_load_config_defaults(tmp_path):
    cfg_file = tmp_path / "minimal.yaml"
    cfg_file.write_text("models: []\n")
    config = load_config(cfg_file)
    assert config.probe_dir == "probes"
    assert config.num_generations == 5
    assert config.toxicity_threshold == 0.5


def test_model_config_defaults():
    mc = ModelConfig(name="test", hf_id="test/model")
    assert mc.max_new_tokens == 40
    assert mc.batch_size == 8
    assert mc.device == "cpu"
