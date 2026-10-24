"""
用固定窗口与提示词对照现有 checkpoint，不训练、不重写模型或语料。
报告多组采样的描述性波动；生成重复率仅作为可复核代理指标，不自动评判故事质量。
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description="Evaluate existing TinyStories checkpoints with fixed windows and prompts.")
    parser.add_argument("--config",type=Path,default=Path("configs/tinystories-cpu.yaml"))
    parser.add_argument("--prompts",type=Path,default=Path("experiments/tinystories-prompts.json"))
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/tinystories-evaluation-v1"))
    args=parser.parse_args()
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    import torch
    from tiny_lm.config import load_config
    from tiny_lm.data import load_token_data
    from tiny_lm.evaluate import evaluate
    from tiny_lm.generate import load_generation_model,generate
    from tiny_lm.checkpoint import read_checkpoint
    from tiny_lm.binding import build_run_state,check_bindings
    from tiny_lm.tokenizer.tokenizer import BPETokenizer
    torch.set_num_threads(1)
    started=time.perf_counter()
    config=load_config(args.config)
    prompts=json.loads(args.prompts.read_text(encoding="utf-8"))
    if len(prompts)!=10 or len({row["id"] for row in prompts})!=10 or any(not row["prompt"].strip() for row in prompts):
        raise ValueError("exactly ten distinct IDs with nonempty prompts are required")
    steps=(250,500,750,1000)
    checkpoints={step:config.output.output_dir/f"step_{step:08d}.pt" for step in steps}
    sources=[args.config.resolve(),args.prompts.resolve(),config.data.tokenizer_path,config.data.train_data_path,config.data.val_data_path,*checkpoints.values()]
    hashes={str(path):fingerprint(path) for path in sources}
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    tokenizer=BPETokenizer.from_file(config.data.tokenizer_path)
    arrays={}
    rows=[]
    samples=[]
    details=[]
    seeds=(1337,2026,42)
    try:
        arrays["train"]=load_token_data(config.data.train_data_path,tokenizer=tokenizer)
        arrays["val"]=load_token_data(config.data.val_data_path,tokenizer=tokenizer)
        binding=build_run_state(arrays["train"],arrays["val"],dtype=torch.float32,eval_batches=4,tokenizer=tokenizer)
        for step,path in checkpoints.items():
            payload=read_checkpoint(path)
            if payload["iteration"]!=step:
                raise ValueError("checkpoint iteration does not match its expected step")
            check_bindings(payload,run_state=binding)
            del payload
            model=load_generation_model(path,device="cpu",config=config.model,tokenizer=tokenizer)
            row={"step":step}
            for split,data in arrays.items():
                losses=[]
                for seed in seeds:
                    result=evaluate(model,data,batch_size=8,context_length=128,num_batches=32,generator=torch.Generator(device="cpu").manual_seed(seed))
                    losses.append(result.loss)
                    details.append({"step":step,"split":split,"seed":seed,"loss":result.loss,"num_tokens":result.num_tokens})
                row[split+"_loss"]=statistics.mean(losses)
                row[split+"_seed_sd"]=statistics.stdev(losses)
                row[split+"_perplexity"]=math.exp(row[split+"_loss"])
            row["val_minus_train"]=row["val_loss"]-row["train_loss"]
            for index,prompt in enumerate(prompts):
                ids=tokenizer.encode(prompt["prompt"])
                generated=generate(model,torch.tensor([ids],dtype=torch.int64),128,eos_token_id=tokenizer.special_to_id["<|endoftext|>"],temperature=0.8,top_p=1.0,top_k=20,generator=torch.Generator(device="cpu").manual_seed(42+index))[0].tolist()
                new=generated[len(ids):]
                eos=bool(new and new[-1]==tokenizer.special_to_id["<|endoftext|>"])
                content=new[:-1] if eos else new
                grams=[tuple(content[i:i+4]) for i in range(max(0,len(content)-3))]
                repetition=0.0 if not grams else 1-len(set(grams))/len(grams)
                samples.append({"step":step,**prompt,"seed":42+index,"text":tokenizer.decode(generated),"token_ids":generated,"new_tokens":len(new),"finish_reason":"eos" if eos else "length","repeated_token_4gram_fraction":repetition})
            current=[sample for sample in samples if sample["step"]==step]
            row["eos_count"]=sum(sample["finish_reason"]=="eos" for sample in current)
            row["mean_new_tokens"]=statistics.mean(sample["new_tokens"] for sample in current)
            row["mean_repeated_token_4gram_fraction"]=statistics.mean(sample["repeated_token_4gram_fraction"] for sample in current)
            rows.append(row)
            print(json.dumps(row),flush=True)
            del model
        if any(fingerprint(Path(path))!=value for path,value in hashes.items()):
            raise ValueError("an input or checkpoint changed during evaluation")
        report={"python":platform.python_version(),"torch":str(torch.__version__),"threads":1,"seeds":seeds,"batches_per_seed":32,"batch_size":8,"context_length":128,"tokens_per_split_per_checkpoint":98304,"generation":{"temperature":0.8,"top_k":20,"top_p":1.0,"max_new_tokens":128,"seed_rule":"42 + prompt index"},"source_sha256":hashes,"run_state":binding,"metrics":rows,"seed_metrics":details,"samples":samples,"elapsed_seconds":time.perf_counter()-started}
        (output/"report.json").write_text(json.dumps(report,ensure_ascii=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        with (output/"metrics.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        markdown=["# TinyStories 固定提示生成全集","","10 条预先固定提示，每个 checkpoint 全部保留；未挑选样本。参数见评估报告。",""]
        for sample in samples:
            markdown.extend([f"## Step {sample['step']} / {sample['id']}","",f"Seed: {sample['seed']}; new tokens: {sample['new_tokens']}; finish: {sample['finish_reason']}","","```text",sample["text"],"```",""])
        (output/"samples.md").write_text("\n".join(markdown),encoding="utf-8")
        print(f"Evaluation complete: {output}; {report['elapsed_seconds']:.2f}s; source/checkpoint hashes unchanged.",flush=True)
    finally:
        for data in arrays.values():
            if getattr(data,"_mmap",None) is not None:
                data._mmap.close()


if __name__=="__main__":
    main()
