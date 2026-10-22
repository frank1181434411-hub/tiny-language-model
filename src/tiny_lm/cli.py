"""
提供配置检查、预处理、模型训练与验证命令；生成入口待接通。
"""

from __future__ import annotations
import argparse
import json
import pickle
import sys
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from tiny_lm import __version__
from tiny_lm.config import ConfigError,load_config

_PENDING="尚未实现，等待源码迁移"


def build_parser()->argparse.ArgumentParser:
    parser=argparse.ArgumentParser(prog="tiny-lm",description="个人语言模型项目：配置检查、预处理、训练与验证。")
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

    train=commands.add_parser("train",help="从 token 文件训练模型或恢复训练",description="读取配置指定的 tokenizer、训练/验证数据；按间隔验证和保存 checkpoint。")
    train.add_argument("--config",type=Path,required=True,help="YAML 配置路径")
    train.add_argument("--resume",type=Path,help="恢复训练所用 checkpoint 路径")
    train.add_argument("--stop-after",type=int,help="本次最多完成的总更新次数；不改变配置中的 max_steps")

    evaluate=commands.add_parser("evaluate",help="加载 checkpoint 计算验证 loss",description="只读取验证数据，使用独立采样器估计平均 token loss；不保存模型或恢复训练随机状态。")
    evaluate.add_argument("--config",type=Path,required=True,help="YAML 配置路径")
    evaluate.add_argument("--checkpoint",type=Path,required=True,help="checkpoint 路径")
    evaluate.add_argument("--seed",type=int,help="独立验证采样种子，默认使用配置中的 training.seed")
    for command in (train,evaluate):
        command.add_argument("--dtype",choices=("float16","bfloat16","float32","float64"),default="float32",help="模型参数精度，默认 float32；恢复或验证时须与 checkpoint 一致")
        command.add_argument("--eval-batches",type=int,default=4,help="每次验证采样 batch 数，必须为正整数，默认 4")

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


def _training_command(args:argparse.Namespace)->dict:
    """
    校验配置、tokenizer 与 token 文件后调用数组接口；退出时关闭只读映射。
    验证复用 checkpoint 的模型与 optimizer 校验，关闭训练随机状态恢复。
    """
    import torch
    from tiny_lm.data import load_token_data
    from tiny_lm.evaluate import _positive_integer,evaluate
    from tiny_lm.tokenizer.tokenizer import BPETokenizer
    config=load_config(args.config)
    _positive_integer("eval_batches",args.eval_batches)
    seed=config.training.seed if args.command=="train" or args.seed is None else args.seed
    if not 0<=seed<2**64:
        raise ValueError("seed must be between 0 and 2**64 - 1")
    device=torch.device(config.training.device)
    if device.type not in ("cpu","cuda"):
        raise ValueError("training and evaluation currently support CPU and CUDA devices only")
    if device.type=="cuda" and (not torch.cuda.is_available() or (device.index is not None and device.index>=torch.cuda.device_count())):
        raise ValueError("the configured CUDA device is not available")
    if args.command=="train" and args.stop_after is not None and not 0<=args.stop_after<=config.training.max_steps:
        raise ValueError("stop_after must be between 0 and config.training.max_steps")
    if config.model.ffn_type!="swiglu":
        raise ValueError("configured ffn_type must be swiglu; GELU is not implemented")
    tokenizer=BPETokenizer.from_file(config.data.tokenizer_path)
    if len(tokenizer.vocab)!=config.model.vocab_size:
        raise ValueError("model.vocab_size must match the actual tokenizer vocabulary size")
    dtype=getattr(torch,args.dtype)
    with ExitStack() as stack:
        def load_data(path):
            data=load_token_data(path,tokenizer=tokenizer)
            mapping=getattr(data,"_mmap",None)
            if mapping is not None:
                stack.callback(mapping.close)
            if len(data)<=config.model.context_length:
                raise ValueError(f"token data must contain at least context_length + 1 tokens: {path}")
            return data
        validation_data=load_data(config.data.val_data_path)
        if args.command=="train":
            from tiny_lm.train import train
            def report(metrics):
                print(json.dumps(asdict(metrics),allow_nan=False),file=sys.stderr,flush=True)
            result=train(config,load_data(config.data.train_data_path),validation_data,dtype=dtype,eval_batches=args.eval_batches,resume_from=args.resume,stop_after=args.stop_after,on_step=report)
            return {"completed_updates":result.completed_updates,"checkpoints":[str(path) for path in result.checkpoints],"last_metrics":asdict(result.history[-1]) if result.history else None,"dtype":args.dtype,"eval_batches":args.eval_batches}
        from tiny_lm.checkpoint import load_checkpoint
        from tiny_lm.model.transformer import TransformerLM
        from tiny_lm.optim import AdamW
        model=TransformerLM.from_config(config.model,device=device,dtype=dtype)
        optimizer=AdamW(model.parameters(),lr=config.training.learning_rate,weight_decay=config.training.weight_decay)
        # 独立验证允许新的验证路径和设备；模型结构仍由 checkpoint 严格校验。
        iteration=load_checkpoint(args.checkpoint,model,optimizer,restore_rng=False)
        del optimizer
        generator=torch.Generator(device="cpu").manual_seed(seed)
        result=evaluate(model,validation_data,batch_size=config.training.batch_size,context_length=config.model.context_length,num_batches=args.eval_batches,generator=generator)
        return {"checkpoint":str(args.checkpoint.resolve()),"completed_updates":iteration,**asdict(result),"seed":seed,"dtype":args.dtype}


def main(argv:list[str]|None=None)->int:
    args=build_parser().parse_args(argv)
    if args.command in ("train-tokenizer","prepare-data","train","evaluate"):
        try:
            result=_preprocess_command(args) if args.command in ("train-tokenizer","prepare-data") else _training_command(args)
        except ImportError as exc:
            extra="compute" if args.command in ("train","evaluate") else "tokenizer" if args.command=="train-tokenizer" else "tokenizer,data"
            print(f"{args.command}: {exc}. Install dependencies with: python -m pip install '.[{extra}]'",file=sys.stderr)
            return 2
        except (OSError,ValueError,TypeError,RuntimeError,FloatingPointError,EOFError,pickle.UnpicklingError) as exc:
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
