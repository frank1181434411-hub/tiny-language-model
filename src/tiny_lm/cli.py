"""Lightweight CLI for config validation and explicitly unavailable future commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tiny_lm import __version__
from tiny_lm.config import ConfigError, load_config

_PENDING = "尚未实现，等待源码迁移"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny-lm",
        description="个人语言模型项目：当前支持配置校验，训练及生成等待源码迁移。",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-config", help="校验 YAML 配置并显示规范化参数及绝对路径")
    check.add_argument("--config", type=Path, required=True, help="YAML 配置路径")

    tokenizer = commands.add_parser(
        "train-tokenizer", help=f"训练 BPE tokenizer（{_PENDING}）", description=_PENDING
    )
    tokenizer.add_argument("--input", type=Path, required=True, help="输入 UTF-8 文本路径")
    tokenizer.add_argument("--vocab-size", type=int, required=True, help="目标词表大小")
    tokenizer.add_argument("--output", type=Path, required=True, help="tokenizer 输出路径")

    prepare = commands.add_parser(
        "prepare-data", help=f"文本编码与数据预处理（{_PENDING}）", description=_PENDING
    )
    prepare.add_argument("--input", type=Path, required=True, help="输入 UTF-8 文本路径")
    prepare.add_argument("--tokenizer", type=Path, required=True, help="tokenizer 路径")
    prepare.add_argument("--output", type=Path, required=True, help="编码数据输出路径")

    train = commands.add_parser("train", help=f"训练模型（{_PENDING}）", description=_PENDING)
    train.add_argument("--config", type=Path, required=True, help="YAML 配置路径")
    train.add_argument("--resume", type=Path, help="恢复训练所用 checkpoint 路径")

    evaluate = commands.add_parser(
        "evaluate", help=f"计算验证 loss（{_PENDING}）", description=_PENDING
    )
    evaluate.add_argument("--config", type=Path, required=True, help="YAML 配置路径")
    evaluate.add_argument("--checkpoint", type=Path, required=True, help="checkpoint 路径")

    generate = commands.add_parser("generate", help=f"文本续写（{_PENDING}）", description=_PENDING)
    generate.add_argument("--config", type=Path, required=True, help="YAML 配置路径")
    generate.add_argument("--checkpoint", type=Path, required=True, help="checkpoint 路径")
    generate.add_argument("--prompt", required=True, help="续写提示文本")
    generate.add_argument("--max-new-tokens", type=int, default=64, help="最大新增 token 数")
    generate.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    generate.add_argument("--top-k", type=int, help="可选 top-k 采样参数")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "check-config":
        print(f"{args.command}: {_PENDING}", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    print("配置校验通过（仅结构与参数校验；未检查数据存在性或硬件可用性）。")
    print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
    return 0
