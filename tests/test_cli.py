"""Exercise the installed module and console entry points in subprocesses."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tiny_lm import __version__

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ["check-config", "train-tokenizer", "prepare-data", "train", "evaluate", "generate"]


@pytest.fixture(params=["module", "console"])
def entry(request):
    if request.param == "module":
        return [sys.executable, "-m", "tiny_lm"]
    executable = shutil.which("tiny-lm", path=str(Path(sys.executable).parent))
    assert executable is not None, "Install the package in this virtual environment first"
    return [executable]


def run_cli(entry, args, cwd):
    return subprocess.run(
        [*entry, *args],
        cwd=cwd,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_help(entry, tmp_path):
    result = run_cli(entry, ["--help"], tmp_path)
    assert result.returncode == 0
    assert "check-config" in result.stdout
    for command in COMMANDS:
        assert command in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_version(entry, tmp_path):
    result = run_cli(entry, ["--version"], tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == f"tiny-lm {__version__}"


def test_check_config(entry, tmp_path):
    # Copy the config so any unintended output creation is confined to this test.
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "smoke.yaml"
    config.write_text(
        (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    before = set(tmp_path.rglob("*"))
    result = run_cli(entry, ["check-config", "--config", str(config)], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "配置校验通过" in result.stdout
    normalized = json.loads(result.stdout.split("\n", 1)[1])
    assert normalized["training"]["device"] == "cpu"
    assert normalized["output"]["output_dir"] == str(tmp_path / "outputs" / "smoke")
    assert normalized["data"]["train_data_path"] == str(tmp_path / "data" / "smoke" / "train.bin")
    assert set(tmp_path.rglob("*")) == before


def test_config_error_is_nonzero(entry, tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("unexpected: true", encoding="utf-8")
    result = run_cli(entry, ["check-config", "--config", str(path)], tmp_path)
    assert result.returncode == 2
    assert "配置错误" in result.stderr
    assert "未知字段" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("command", COMMANDS)
def test_required_arguments_fail_without_output(entry, tmp_path, command):
    result = run_cli(entry, [command], tmp_path)
    assert result.returncode == 2
    assert "required" in result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help(entry, tmp_path, command):
    result = run_cli(entry, [command, "--help"], tmp_path)
    assert result.returncode == 0
    assert "尚未实现" not in result.stdout
