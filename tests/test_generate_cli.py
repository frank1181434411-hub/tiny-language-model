"""
生成命令的真实 checkpoint、tokenizer、输出与错误处理测试。
"""

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
import yaml
from tiny_lm.cli import main
from tiny_lm.config import load_config
from tiny_lm.checkpoint import save_checkpoint
from tiny_lm.tokenizer.tokenizer import BPETokenizer
from tiny_lm.preprocess import process_corpus
from tiny_lm.binding import build_run_state

torch=pytest.importorskip("torch")
np=pytest.importorskip("numpy")
pytest.importorskip("regex")
TransformerLM=importlib.import_module("tiny_lm.model.transformer").TransformerLM
AdamW=importlib.import_module("tiny_lm.optim").AdamW


@pytest.fixture(autouse=True)
def preserve_rng():
    state=torch.get_rng_state()
    yield
    torch.set_rng_state(state)


@pytest.fixture(params=["module","console"])
def entry(request):
    if request.param=="module":
        return [sys.executable,"-m","tiny_lm"]
    executable=shutil.which("tiny-lm",path=str(Path(sys.executable).parent))
    assert executable is not None
    return [executable]


def run_cli(entry,args,cwd):
    return subprocess.run([*entry,*args],cwd=cwd,env={**os.environ,"PYTHONIOENCODING":"utf-8","PYTHONDONTWRITEBYTECODE":"1"},capture_output=True,text=True,encoding="utf-8",check=False)


def write_checkpoint(config,path,target=122,dtype=torch.float64):
    model=TransformerLM.from_config(config.model,dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.token_embeddings.weight.fill_(1)
        model.ln_final.weight.fill_(1)
        model.lm_head.weight[target].fill_(1)
    tokenizer=BPETokenizer.from_file(config.data.tokenizer_path)
    data=np.arange(16,dtype=np.int64)
    run_state=build_run_state(data,data,dtype=dtype,eval_batches=4,tokenizer=tokenizer)
    save_checkpoint(model,AdamW(model.parameters()),0,path,config=config,run_state=run_state)


@pytest.fixture
def project(tmp_path):
    directory=tmp_path/"project with spaces"
    directory.mkdir()
    tokenizer=BPETokenizer({**{i:bytes([i]) for i in range(256)},256:b"<E>",257:b"<|endoftext|>"},[],["<E>","<|endoftext|>"])
    tokenizer.save(directory/"tokenizer.json")
    options={"model":{"vocab_size":258,"context_length":4,"d_model":8,"num_layers":1,"num_heads":2,"d_ff":16,"norm_mode":"pre_norm","ffn_type":"swiglu","rope_theta":10000.0},"data":{"train_data_path":"missing-train.bin","val_data_path":"missing-val.bin","tokenizer_path":"tokenizer.json"},"training":{"batch_size":2,"max_steps":1,"learning_rate":0.01,"min_learning_rate":0.001,"warmup_steps":0,"weight_decay":0.01,"grad_clip":1.0,"seed":42,"device":"cpu","eval_interval":1,"save_interval":1},"output":{"output_dir":"unused-output"}}
    path=directory/"run.yaml"
    path.write_text(yaml.safe_dump(options),encoding="utf-8")
    write_checkpoint(load_config(path),directory/"model.pt")
    return path


def arguments(project):
    return ["generate","--config",str(project),"--checkpoint",str(project.parent/"model.pt"),"--prompt","你好🙂 long prompt","--max-new-tokens","3","--top-k","1"]


def test_console_generation_and_seed_reproducibility(entry,project,tmp_path):
    before={p.name:p.read_bytes() for p in project.parent.iterdir()}
    result=run_cli(entry,arguments(project),tmp_path)
    assert result.returncode==0,result.stderr
    assert result.stderr==""
    report=json.loads(result.stdout)
    assert report["text"]=="你好🙂 long promptzzz"
    assert report["num_generated_tokens"]==3
    assert report["num_prompt_tokens"]>4
    assert report["dtype"]=="float64"
    assert report["finish_reason"]=="length"
    stochastic=[*arguments(project),"--top-k","20","--top-p","0.9","--temperature","1.4","--seed","123"]
    first=run_cli(entry,stochastic,tmp_path)
    second=run_cli(entry,stochastic,tmp_path)
    assert first.returncode==second.returncode==0
    assert first.stdout==second.stdout
    assert {p.name:p.read_bytes() for p in project.parent.iterdir()}==before


@pytest.mark.parametrize("eos,token_id",[(None,257),("<E>",256)])
def test_eos_and_prompt_special_modes(project,capsys,eos,token_id):
    write_checkpoint(load_config(project),project.parent/"model.pt",target=token_id)
    extra=[] if eos is None else ["--eos-token",eos]
    assert main([*arguments(project),"--prompt","<E>",*extra])==0
    report=json.loads(capsys.readouterr().out)
    assert report["num_generated_tokens"]==1
    assert report["finish_reason"]=="eos"
    assert report["token_ids"]==[256,token_id]
    assert main([*arguments(project),"--prompt","<E>","--special-mode","ordinary",*extra])==0
    report=json.loads(capsys.readouterr().out)
    assert report["token_ids"]==[60,69,62,token_id]


def test_zero_tokens_preserves_full_unicode_prompt(project,capsys):
    assert main([*arguments(project),"--max-new-tokens","0"])==0
    report=json.loads(capsys.readouterr().out)
    assert report["text"]=="你好🙂 long prompt"
    assert report["num_generated_tokens"]==0
    assert report["finish_reason"]=="length"


@pytest.mark.parametrize("extra",[["--prompt",""],["--temperature","0"],["--temperature","nan"],["--temperature","inf"],["--top-p","0"],["--top-p","1.1"],["--top-k","0"],["--top-k","259"],["--max-new-tokens","-1"],["--seed","-1"],["--seed",str(2**64)],["--eos-token","<missing>"]])
def test_bad_options_produce_readable_errors(project,capsys,extra):
    assert main([*arguments(project),*extra])==2
    result=capsys.readouterr()
    assert result.out==""
    assert result.err.startswith("generate:")
    assert "Traceback" not in result.err
    assert not (project.parent/"unused-output").exists()


@pytest.mark.parametrize("problem",["checkpoint","tokenizer","vocab","model","device"])
def test_incompatible_inputs(project,capsys,problem):
    if problem=="checkpoint":
        (project.parent/"model.pt").write_bytes(b"broken checkpoint")
    elif problem=="tokenizer":
        (project.parent/"tokenizer.json").write_text("{",encoding="utf-8")
    else:
        options=yaml.safe_load(project.read_text(encoding="utf-8"))
        if problem=="vocab":
            options["model"]["vocab_size"]=259
        elif problem=="model":
            options["model"]["norm_mode"]="post_norm"
        else:
            options["training"]["device"]="mps"
        project.write_text(yaml.safe_dump(options),encoding="utf-8")
    assert main(arguments(project))==2
    result=capsys.readouterr()
    assert result.out==""
    assert "Traceback" not in result.err
    assert not (project.parent/"unused-output").exists()


def test_real_file_training_checkpoint_can_generate(project,capsys):
    config=load_config(project)
    tokenizer=BPETokenizer.from_file(config.data.tokenizer_path)
    for name,path in (("train",config.data.train_data_path),("val",config.data.val_data_path)):
        source=project.parent/f"{name}.txt"
        source.write_text(("abc cab " if name=="train" else "cab abc ")*20,encoding="utf-8")
        process_corpus(source,path,tokenizer)
    assert main(["train","--config",str(project),"--eval-batches","1"])==0
    trained=json.loads(capsys.readouterr().out)
    assert main([*arguments(project),"--checkpoint",trained["checkpoints"][-1]])==0
    report=json.loads(capsys.readouterr().out)
    assert report["dtype"]=="float32"
    assert report["text"].startswith("你好🙂 long prompt")
    assert 1<=report["num_generated_tokens"]<=3
