"""
通过真实 CLI 完成小样本训练、中断续训、独立验证与生成，保存可核对的实验记录。
另跑同配置连续训练，对照全部权重、优化器、采样器状态和每步指标。
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def equal(left,right):
    import torch
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(equal(left[key],right[key]) for key in left)
    if isinstance(left,(list,tuple)):
        return type(left) is type(right) and len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right,strict=True))
    return left==right


def main():
    parser=argparse.ArgumentParser(description="Run the 50-step CPU TinyStories pilot and exact resume comparison.")
    parser.add_argument("--config",type=Path,default=Path("configs/tinystories-pilot.yaml"))
    args=parser.parse_args()
    # 仅影响本进程与子进程，不改变系统环境或项目默认训练算法。
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    import torch
    from tiny_lm.config import load_config
    from tiny_lm.data import load_token_data
    from tiny_lm.evaluate import evaluate
    from tiny_lm.train import train
    from tiny_lm.checkpoint import read_checkpoint
    from tiny_lm.tokenizer.tokenizer import BPETokenizer
    torch.set_num_threads(1)
    config_path=args.config.resolve(strict=True)
    config=load_config(config_path)
    if config.training.device!="cpu" or config.training.max_steps!=50:
        raise ValueError("this pilot requires a CPU config with max_steps=50")
    output=config.output.output_dir
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"config.resolved.json",config.to_dict())
    (output/"config.source.yaml").write_bytes(config_path.read_bytes())
    report={"python":platform.python_version(),"platform":platform.platform(),"processor":platform.processor(),"logical_cpus":os.cpu_count(),"torch":str(torch.__version__),"cuda_available":torch.cuda.is_available(),"torch_threads":torch.get_num_threads(),"dtype":"float32","eval_batches_training":4,"eval_seed":1337,"eval_batches_independent":16,"commands":[]}
    git=subprocess.run(["git","-c",f"safe.directory={Path.cwd().as_posix()}","rev-parse","HEAD"],capture_output=True,text=True,check=False)
    report["git_revision"]=git.stdout.strip() if git.returncode==0 else None
    status=subprocess.run(["git","-c",f"safe.directory={Path.cwd().as_posix()}","status","--short"],capture_output=True,text=True,check=False)
    report["working_tree_status"]=status.stdout

    def cli(name,arguments):
        command=[sys.executable,"-m","tiny_lm",*arguments]
        print(f"Running {name}...",flush=True)
        started=time.perf_counter()
        with (output/(name+".json")).open("w",encoding="utf-8") as stdout,(output/(name+".stderr.jsonl")).open("w",encoding="utf-8") as stderr:
            result=subprocess.run(command,stdout=stdout,stderr=stderr,check=False)
        elapsed=time.perf_counter()-started
        report["commands"].append({"name":name,"argv":command,"wall_seconds":elapsed,"returncode":result.returncode})
        if result.returncode!=0:
            raise RuntimeError(f"{name} failed; inspect {output/(name+'.stderr.jsonl')}")
        print(f"{name} completed in {elapsed:.2f}s.",flush=True)
        return json.loads((output/(name+".json")).read_text(encoding="utf-8"))

    tokenizer=BPETokenizer.from_file(config.data.tokenizer_path)
    train_data=load_token_data(config.data.train_data_path,tokenizer=tokenizer)
    validation_data=None
    try:
        validation_data=load_token_data(config.data.val_data_path,tokenizer=tokenizer)
        initial=train(config,train_data,validation_data,stop_after=0,save_checkpoints=False,tokenizer=tokenizer)
        report["parameter_count"]=sum(p.numel() for p in initial.model.parameters())
        report["initial_evaluation"]=asdict(evaluate(initial.model,validation_data,batch_size=config.training.batch_size,context_length=config.model.context_length,num_batches=16,generator=torch.Generator(device="cpu").manual_seed(1337)))
        del initial
        cli("train-first-25",["train","--config",str(config_path),"--stop-after","25","--dtype","float32","--eval-batches","4"])
        checkpoint25=output/"step_00000025.pt"
        checkpoint50=output/"step_00000050.pt"
        cli("train-resume-50",["train","--config",str(config_path),"--resume",str(checkpoint25),"--dtype","float32","--eval-batches","4"])
        report["final_evaluation"]=cli("evaluate-final",["evaluate","--config",str(config_path),"--checkpoint",str(checkpoint50),"--dtype","float32","--eval-batches","16","--seed","1337"])
        report["generation"]=cli("generate-final",["generate","--config",str(config_path),"--checkpoint",str(checkpoint50),"--prompt","Once upon a time, there was a little girl","--max-new-tokens","64","--temperature","0.8","--top-k","20","--seed","42"])
        print("Checking uninterrupted 50-step reference...",flush=True)
        started=time.perf_counter()
        reference=train(config,train_data,validation_data,save_checkpoints=False,tokenizer=tokenizer)
        report["reference_wall_seconds"]=time.perf_counter()-started
        payload=read_checkpoint(checkpoint50)
        history=[]
        for name in ("train-first-25","train-resume-50"):
            history.extend(json.loads(line) for line in (output/(name+".stderr.jsonl")).read_text(encoding="utf-8").splitlines() if line.strip())
        checks={"model":equal(payload["model_state_dict"],reference.model.state_dict()),"optimizer":equal(payload["optimizer_state_dict"],reference.optimizer.state_dict()),"history":history==[asdict(row) for row in reference.history],"generators":all(torch.equal(payload["rng_state"]["generators"][name],generator.get_state()) for name,generator in reference.generators.items())}
        report["resume_exact_checks"]=checks
        if not all(checks.values()):
            write_json(output/"report.failed.json",report)
            raise RuntimeError(f"uninterrupted and resumed runs differ: {checks}")
        report["run_state"]=payload["run_state"]
        report["training_tokens"]=50*config.training.batch_size*config.model.context_length
        report["training_cli_wall_seconds"]=sum(command["wall_seconds"] for command in report["commands"] if command["name"].startswith("train-"))
        report["training_cli_tokens_per_second"]=report["training_tokens"]/report["training_cli_wall_seconds"]
        report["training_history"]=history
        write_json(output/"report.json",report)
        print(json.dumps({"output_dir":str(output),"parameter_count":report["parameter_count"],"initial_loss":report["initial_evaluation"]["loss"],"final_loss":report["final_evaluation"]["loss"],"train_wall_seconds":report["training_cli_wall_seconds"],"resume_exact_checks":checks}),flush=True)
    finally:
        for data in (train_data,validation_data):
            if getattr(data,"_mmap",None) is not None:
                data._mmap.close()


if __name__=="__main__":
    main()
