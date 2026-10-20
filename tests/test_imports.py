"""Import every package module in a fresh interpreter with file-write auditing."""

import os
import subprocess
import sys
from pathlib import Path


def test_all_modules_import_without_side_effects(tmp_path):
    package = Path(__file__).resolve().parents[1] / "src" / "tiny_lm"
    modules = []
    for path in sorted(package.rglob("*.py")):
        parts = list(path.relative_to(package.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules.append(".".join(parts))
    script = """
import importlib
import os
import sys

def audit(event, args):
    if event == "open":
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError(f"Import attempted file write: {path}")
        if isinstance(path, str) and path.endswith((".bin", ".yaml", ".json", ".txt", ".pt")):
            raise AssertionError(f"Import attempted data read: {path}")
    if event in ("os.mkdir", "os.remove", "os.rename", "subprocess.Popen", "socket.connect"):
        raise AssertionError(f"Import attempted side effect: {event}")

sys.addaudithook(audit)
for module in sys.argv[1:]:
    importlib.import_module(module)
assert not ({"torch", "numpy", "regex"} & sys.modules.keys())
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, *modules],
        cwd=tmp_path,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []
