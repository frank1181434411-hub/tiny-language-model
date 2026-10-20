"""Strict dataclass/YAML configuration; no data access or output-directory creation."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


class ConfigError(ValueError):
    """A configuration could not be read or violates the scaffold schema."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently replacing values."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError("配置字段名必须是字符串")
            if key in result:
                raise ConfigError(f"重复配置字段: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    norm_mode: str
    ffn_type: str
    rope_theta: float

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context_length", "num_layers", "d_model", "num_heads", "d_ff"):
            _integer(f"model.{name}", getattr(self, name), minimum=1)
        _enum("model.norm_mode", self.norm_mode, ("pre_norm", "post_norm"))
        _enum("model.ffn_type", self.ffn_type, ("swiglu", "gelu"))
        _number("model.rope_theta", self.rope_theta, positive=True)
        if self.d_model % self.num_heads:
            raise ConfigError("model.d_model 必须能被 model.num_heads 整除")
        if (self.d_model // self.num_heads) % 2:
            raise ConfigError("RoPE 的 head dimension (d_model / num_heads) 必须为偶数")


@dataclass(frozen=True)
class DataConfig:
    train_data_path: Path
    val_data_path: Path
    tokenizer_path: Path

    def __post_init__(self) -> None:
        for field in fields(self):
            _path(f"data.{field.name}", getattr(self, field.name))


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    max_steps: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    seed: int
    device: str
    eval_interval: int
    save_interval: int

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_steps", "eval_interval", "save_interval"):
            _integer(f"training.{name}", getattr(self, name), minimum=1)
        for name in ("warmup_steps", "seed"):
            _integer(f"training.{name}", getattr(self, name), minimum=0)
        _number("training.learning_rate", self.learning_rate, positive=True)
        _number("training.min_learning_rate", self.min_learning_rate, positive=False)
        _number("training.weight_decay", self.weight_decay, positive=False)
        _number("training.grad_clip", self.grad_clip, positive=True)
        if self.min_learning_rate > self.learning_rate:
            raise ConfigError("training.min_learning_rate 不得大于 learning_rate")
        if self.warmup_steps >= self.max_steps:
            raise ConfigError("training.warmup_steps 必须小于 max_steps，且不得为负")
        if not isinstance(self.device, str) or not re.fullmatch(
            r"cpu|mps|cuda(?::[0-9]+)?", self.device
        ):
            raise ConfigError("training.device 必须是 cpu、mps、cuda 或 cuda:N（N 为非负整数）")


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path

    def __post_init__(self) -> None:
        _path("output.output_dir", self.output_dir)


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible view, including resolved paths."""
        result = asdict(self)
        for section in ("data", "output"):
            result[section] = {name: str(value) for name, value in result[section].items()}
        return result


def _integer(name: str, value: Any, *, minimum: int) -> None:
    # bool is an int subclass, but is never a valid integer configuration value.
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{name} 必须是 >= {minimum} 的整数（不接受布尔值）")


def _number(name: str, value: Any, *, positive: bool) -> None:
    if type(value) not in (int, float):
        raise ConfigError(f"{name} 必须是有限数值")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (value <= 0 if positive else value < 0):
        bound = "> 0" if positive else ">= 0"
        raise ConfigError(f"{name} 必须是有限数值且 {bound}")


def _enum(name: str, value: Any, choices: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{name} 必须是以下之一: {', '.join(choices)}")


def _path(name: str, value: Any) -> None:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ConfigError(f"{name} 必须是解析后的绝对 pathlib.Path")


def _mapping(value: Any, name: str, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} 必须是 YAML 映射")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{name} 字段名必须是字符串")
    unknown = set(value) - expected
    if unknown:
        raise ConfigError(f"{name} 包含未知字段: {', '.join(sorted(unknown))}")
    missing = expected - set(value)
    if missing:
        raise ConfigError(f"{name} 缺少必填字段: {', '.join(sorted(missing))}")
    return value


_Section = TypeVar("_Section", ModelConfig, DataConfig, TrainingConfig, OutputConfig)


def _section(cls: type[_Section], raw: Any, name: str, base_dir: Path) -> _Section:
    values = dict(_mapping(raw, name, {field.name for field in fields(cls)}))
    if cls in (DataConfig, OutputConfig):
        for key, value in values.items():
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ConfigError(f"{name}.{key} 必须是非空路径字符串，且不能包含空字符")
            path = Path(value)
            # No environment-variable or ~ expansion; relative paths use the YAML directory.
            values[key] = (base_dir / path).resolve()
    return cls(**values)


def load_config(config_path: str | Path) -> ProjectConfig:
    """Read UTF-8 YAML and resolve all data/output paths relative to the YAML file.

    The CLI's --config argument itself follows the caller's working directory.
    Referenced data/tokenizer/output paths need not exist. Nothing is created.
    """
    try:
        path = Path(config_path).resolve()
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.load(stream, Loader=_UniqueKeyLoader)
        root = _mapping(raw, "config", {"model", "data", "training", "output"})
        return ProjectConfig(
            model=_section(ModelConfig, root["model"], "model", path.parent),
            data=_section(DataConfig, root["data"], "data", path.parent),
            training=_section(TrainingConfig, root["training"], "training", path.parent),
            output=_section(OutputConfig, root["output"], "output", path.parent),
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"无法读取配置 {config_path}: {exc}") from exc
