"""
在预留数据上比较由各自验证集选出的 checkpoint，不根据留出结果挑选模型。
仅加载模型评估；不续训，也不把留出数据冒充 checkpoint 的历史训练输入。
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time


def fingerprint(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description="Compare validation-selected TinyStories checkpoints on untouched held-out data.")
    parser.add_argument("--reports",type=Path,nargs="+",required=True)
    parser.add_argument("--heldout",type=Path,required=True)
    parser.add_argument("--tokenizer",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output already exists; choose a new path")
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    import torch
    from tiny_lm.data import load_token_data
    from tiny_lm.evaluate import evaluate
    from tiny_lm.generate import load_generation_model
    from tiny_lm.tokenizer.tokenizer import BPETokenizer
    torch.set_num_threads(1)
    started=time.perf_counter()
    tokenizer=BPETokenizer.from_file(args.tokenizer)
    selected=[]
    sources={str(p.resolve()):fingerprint(p) for p in [*args.reports,args.heldout,args.tokenizer]}
    # 所有候选先通过历史验证 loss 确定，然后才读取留出数据计算指标。
    for path in args.reports:
        report=json.loads(path.read_text(encoding="utf-8"))
        if report["run_state"]["dtype"]!="float32":
            raise ValueError("this comparison requires FP32 checkpoints")
        for source,expected in report["source_sha256"].items():
            if fingerprint(Path(source))!=expected:
                raise ValueError(f"evaluation source changed: {source}")
        best=min(report["metrics"],key=lambda row:(row["val_loss"],row["step"]))
        checkpoints=[Path(p) for p in report["source_sha256"] if Path(p).name==f"step_{best['step']:08d}.pt"]
        if len(checkpoints)!=1:
            raise ValueError("expected one checkpoint matching validation-selected step")
        checkpoint=checkpoints[0]
        sources[str(checkpoint)]=fingerprint(checkpoint)
        selected.append({"evaluation_report":str(path.resolve()),"checkpoint":str(checkpoint),"step":best["step"],"selection_val_loss":best["val_loss"]})
    seeds=(1337,2026,42)
    data=load_token_data(args.heldout,tokenizer=tokenizer)
    try:
        for row in selected:
            model=load_generation_model(Path(row["checkpoint"]),device="cpu",tokenizer=tokenizer)
            details=[]
            for seed in seeds:
                result=evaluate(model,data,batch_size=8,context_length=128,num_batches=32,generator=torch.Generator(device="cpu").manual_seed(seed))
                details.append({"seed":seed,"loss":result.loss,"num_tokens":result.num_tokens})
            row["seed_metrics"]=details
            row["loss"]=statistics.mean(item["loss"] for item in details)
            row["seed_sd"]=statistics.stdev(item["loss"] for item in details)
            row["perplexity"]=math.exp(row["loss"])
            print(json.dumps(row),flush=True)
            del model
        if any(fingerprint(Path(path))!=expected for path,expected in sources.items()):
            raise ValueError("a source changed during held-out evaluation")
        result={"python":platform.python_version(),"torch":str(torch.__version__),"threads":1,"selection":"minimum historical validation loss; tie broken by earliest step; heldout not used for selection","batch_size":8,"context_length":128,"batches_per_seed":32,"seeds":seeds,"tokens_per_checkpoint":98304,"heldout_num_tokens":len(data),"source_sha256":sources,"models":selected,"elapsed_seconds":time.perf_counter()-started}
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    finally:
        data._mmap.close()


if __name__=="__main__":
    main()
