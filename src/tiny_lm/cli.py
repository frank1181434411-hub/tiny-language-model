"""
提供配置检查、BPE 训练和文本预处理命令；训练、验证和生成入口待接通。
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from tiny_lm import __version__
from tiny_lm.config import ConfigError,load_config

_PENDING="尚未实现，等待源码迁移"


def build_parser()->argparse.ArgumentParser:
    parser=argparse.ArgumentParser(prog="tiny-lm",description="个人语言模型项目：配置检查、BPE 训练与文本预处理。")
    parser.add_argument("--version",action="version",version=f"%(prog)s {__version__}")
    commands=parser.add_subparsers(dest="command",required=True)

    check=commands.add_parser("check-config",help="校验 YAML 配置并显示规范化参数及绝对路径")
    check.add_argument("--config",type=Path,required=True,help="YAML 配置路径")

    tokenizer=commands.add_parser("train-tokenizer",help="训练并保存 BPE tokenizer",description="读取单个 UTF-8 训练文本，保存原生 v1 JSON；父目录须已存在，不覆盖已有文件。")
    tokenizer.add_argument("--input",type=Path,required=True,help="输入 UTF-8 训练文本路径")
    tokenizer.add_argument("--vocab-size",type=int,required=True,help="目标词表大小；实际大小可能更小")
    tokenizer.add_argument("--output",type=Path,required=True,help="tokenizer JSON 输出路径")
    tokenizer.add_argument("--special-token",action="append",default=[],help="注册特殊 token，可重复指定；默认不注册，ID 按指定顺序分配")

    prepare=commands.add_parser("prepare-data",help="文本编码与数据预处理",description="输出 token 数据及同名 .json 元信息；父目录须已存在，任一输出存在时拒绝覆盖。")
    prepare.add_argument("--input",type=Path,action="append",required=True,help="UTF-8 文本路径，可重复指定多个文件，按参数顺序处理")
    prepare.add_argument("--tokenizer",type=Path,required=True,help="原生 v1 tokenizer JSON 路径")
    prepare.add_argument("--output",type=Path,required=True,help="编码数据输出路径")
    prepare.add_argument("--record-mode",choices=("file","line"),default="file",help="记录边界：默认每个文件一条记录；line 表示每行一条")
    prepare.add_argument("--special-mode",choices=("recognize","ordinary"),default="recognize",help="默认识别特殊 token；ordinary 按普通文本编码")
    prepare.add_argument("--eos-token",help="每条记录末尾追加的已注册特殊 token，默认不追加")
    prepare.add_argument("--write-batch-size",type=int,default=1_000_000,help="每批写盘 token 数，必须为正整数；不限制编码内存")

    train=commands.add_parser("train",help=f"训练模型（{_PENDING}）",description=_PENDING)
    train.add_argument("--config",type=Path,required=True,help="YAML 配置路径")
    train.add_argument("--resume",type=Path,help="恢复训练所用 checkpoint 路径")

    evaluate=commands.add_parser("evaluate",help=f"计算验证 loss（{_PENDING}）",description=_PENDING)
    evaluate.add_argument("--config",type=Path,required=True,help="YAML 配置路径")
    evaluate.add_argument("--checkpoint",type=Path,required=True,help="checkpoint 路径")

    generate=commands.add_parser("generate",help=f"文本续写（{_PENDING}）",description=_PENDING)
    generate.add_argument("--config",type=Path,required=True,help="YAML 配置路径")
    generate.add_argument("--checkpoint",type=Path,required=True,help="checkpoint 路径")
    generate.add_argument("--prompt",required=True,help="续写提示文本")
    generate.add_argument("--max-new-tokens",type=int,default=64,help="最大新增 token 数")
    generate.add_argument("--temperature",type=float,default=1.0,help="采样温度")
    generate.add_argument("--top-k",type=int,help="可选 top-k 采样参数")
    return parser


def _preprocess_command(args:argparse.Namespace)->dict:
    """
    按命令延迟导入文本处理模块，复用现有算法与输出保护。
    """
    if args.command=="train-tokenizer":
        from tiny_lm.tokenizer.bpe import train_bpe
        from tiny_lm.tokenizer.serialization import save_tokenizer
        # 训练前检查目标，避免完成耗时统计后才发现无法输出；保存仍独占创建。
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"output already exists: {args.output}")
        if not args.output.parent.is_dir():
            raise FileNotFoundError(f"output parent directory does not exist: {args.output.parent}")
        vocab,merges=train_bpe(args.input,args.vocab_size,args.special_token)
        save_tokenizer(args.output,vocab,merges,args.special_token)
        return {"output":str(args.output.resolve()),"vocab_size":len(vocab),"num_merges":len(merges),"special_tokens":args.special_token}
    from tiny_lm.preprocess import process_corpus
    from tiny_lm.tokenizer.tokenizer import BPETokenizer
    tokenizer=BPETokenizer.from_file(args.tokenizer)
    output=process_corpus(args.input,args.output,tokenizer,record_mode=args.record_mode,special_mode=args.special_mode,eos_token=args.eos_token,write_batch_size=args.write_batch_size)
    return {"output":str(output),"metadata":str(output.with_name(output.name+".json")),"vocab_size":len(tokenizer.vocab)}


def main(argv:list[str]|None=None)->int:
    args=build_parser().parse_args(argv)
    if args.command in ("train-tokenizer","prepare-data"):
        try:
            result=_preprocess_command(args)
        except ImportError as exc:
            extra="tokenizer" if args.command=="train-tokenizer" else "tokenizer,data"
            print(f"{args.command}: {exc}. Install dependencies with: python -m pip install '.[{extra}]'",file=sys.stderr)
            return 2
        except (OSError,ValueError,TypeError) as exc:
            print(f"{args.command}: {exc}",file=sys.stderr)
            return 2
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    if args.command!="check-config":
        print(f"{args.command}: {_PENDING}",file=sys.stderr)
        return 2
    try:
        config=load_config(args.config)
    except ConfigError as exc:
        print(f"配置错误: {exc}",file=sys.stderr)
        return 2
    print("配置校验通过（仅结构与参数校验；未检查数据存在性或硬件可用性）。")
    print(json.dumps(config.to_dict(),ensure_ascii=False,indent=2))
    return 0
