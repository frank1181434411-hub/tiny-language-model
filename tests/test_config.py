"""Tests for the implemented YAML schema and path resolution only."""

from pathlib import Path

import pytest
import yaml

from tiny_lm.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw_config():
    return yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))


def write_config(tmp_path, raw):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["smoke.yaml", "baseline.example.yaml"])
def test_example_configs(name):
    config = load_config(ROOT / "configs" / name)
    assert config.model.d_model % config.model.num_heads == 0
    assert config.data.train_data_path.is_absolute()
    assert config.output.output_dir.is_absolute()


def test_relative_paths_use_config_directory(tmp_path, raw_config, monkeypatch):
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    raw_config["data"] = {
        "train_data_path": "../missing/train.bin",
        "val_data_path": "nested/val.bin",
        "tokenizer_path": "tokenizer.json",
    }
    raw_config["output"]["output_dir"] = "../new-output"
    path = write_config(config_dir, raw_config)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    config = load_config(path)

    assert config.data.train_data_path == tmp_path / "missing" / "train.bin"
    assert config.data.val_data_path == config_dir / "nested" / "val.bin"
    assert config.data.tokenizer_path == config_dir / "tokenizer.json"
    assert config.output.output_dir == tmp_path / "new-output"
    assert not config.data.train_data_path.exists()
    assert not config.output.output_dir.exists()
    assert config.to_dict()["output"]["output_dir"] == str(tmp_path / "new-output")


def test_absolute_paths_are_preserved(tmp_path, raw_config):
    for key in raw_config["data"]:
        raw_config["data"][key] = str(tmp_path / key)
    raw_config["output"]["output_dir"] = str(tmp_path / "output")
    config = load_config(write_config(tmp_path, raw_config))
    assert config.data.tokenizer_path == tmp_path / "tokenizer_path"
    assert config.output.output_dir == tmp_path / "output"


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("model", "d_model", 30, "divisible"),
        ("model", "d_model", 12, "even"),
        ("model", "num_heads", 0, "integer"),
        ("model", "vocab_size", -1, "integer"),
        ("model", "context_length", True, "integer"),
        ("model", "num_layers", 1.5, "integer"),
        ("model", "d_ff", "64", "integer"),
        ("model", "norm_mode", "typo", "norm_mode"),
        ("model", "ffn_type", "relu", "ffn_type"),
        ("model", "rope_theta", 0, "rope_theta"),
        ("model", "rope_theta", float("inf"), "finite"),
        ("training", "learning_rate", 0, "learning_rate"),
        ("training", "learning_rate", float("nan"), "finite"),
        ("training", "learning_rate", True, "finite"),
        ("training", "min_learning_rate", -0.1, "min_learning_rate"),
        ("training", "min_learning_rate", 0.5, "must not exceed"),
        ("training", "warmup_steps", 4, "less than training.max_steps"),
        ("training", "warmup_steps", -1, "integer"),
        ("training", "max_steps", 0, "integer"),
        ("training", "batch_size", 0, "integer"),
        ("training", "eval_interval", 0, "integer"),
        ("training", "save_interval", False, "integer"),
        ("training", "seed", -1, "integer"),
        ("training", "weight_decay", -0.1, "weight_decay"),
        ("training", "grad_clip", 0, "grad_clip"),
        ("training", "device", "gpu", "device"),
        ("training", "device", "cuda:-1", "device"),
        ("data", "train_data_path", "", "path string"),
        ("data", "tokenizer_path", None, "path string"),
        ("output", "output_dir", False, "path string"),
    ],
)
def test_invalid_values(tmp_path, raw_config, section, field, value, message):
    raw_config[section][field] = value
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, raw_config))


@pytest.mark.parametrize("section", [None, "model", "data", "training", "output"])
def test_unknown_fields_are_rejected(tmp_path, raw_config, section):
    target = raw_config if section is None else raw_config[section]
    target["typo"] = 1
    with pytest.raises(ConfigError, match="unknown fields.*typo"):
        load_config(write_config(tmp_path, raw_config))


def test_missing_fields_are_rejected(tmp_path, raw_config):
    del raw_config["training"]["seed"]
    with pytest.raises(ConfigError, match="missing required fields.*seed"):
        load_config(write_config(tmp_path, raw_config))


@pytest.mark.parametrize("raw", [None, [], {"model": []}, {1: "not a string"}])
def test_invalid_structure(tmp_path, raw):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, raw))


@pytest.mark.parametrize(
    "content", ["model: [", "model: {}\nmodel: {}\n", "model:\n  d_model: 32\n  d_model: 64\n"]
)
def test_malformed_or_duplicate_yaml(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="cannot read configuration"):
        load_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "cuda:0", "cuda:12"])
def test_valid_boundaries_and_devices(tmp_path, raw_config, device):
    raw_config["training"].update(
        max_steps=1, warmup_steps=0, min_learning_rate=0, weight_decay=0, seed=0, device=device
    )
    assert load_config(write_config(tmp_path, raw_config)).training.device == device
