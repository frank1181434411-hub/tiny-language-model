"""
验证文件训练、独立验证、命令入口和错误时的资源释放。
"""

import importlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
import yaml
from tiny_lm.cli import main
from tiny_lm.config import load_config
from tiny_lm.tokenizer.tokenizer import BPETokenizer
from tiny_lm.checkpoint import read_checkpoint
from tiny_lm.data import load_token_data
from tiny_lm.evaluate import evaluate
from tiny_lm.preprocess import process_corpus
from tiny_lm.train import train

torch=pytest.importorskip("torch",reason="training CLI tests require compute dependencies")
np=pytest.importorskip("numpy")
pytest.importorskip("regex")


@pytest.fixture(autouse=True)
def preserve_rng():
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    yield
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


@pytest.fixture(params=["module","console"])
def entry(request):
    if request.param=="module":
        return [sys.executable,"-m","tiny_lm"]
    executable=shutil.which("tiny-lm",path=str(Path(sys.executable).parent))
    assert executable is not None
    return [executable]


def run_cli(entry,args,cwd):
    return subprocess.run([*entry,*args],cwd=cwd,env={**os.environ,"PYTHONIOENCODING":"utf-8","PYTHONDONTWRITEBYTECODE":"1"},capture_output=True,text=True,encoding="utf-8",check=False)


@pytest.fixture
def project(tmp_path):
    folder=tmp_path/"project with spaces"
    folder.mkdir()
    tokenizer=BPETokenizer({**{i:bytes([i]) for i in range(256)},256:b"ab",257:b"<E>"},[(b"a",b"b")],["<E>"])
    tokenizer.save(folder/"tokenizer.json")
    for name,text in (("train","ab ab cab "*20),("val","cab ab "*20)):
        source=folder/f"{name}.txt"
        source.write_text(text,encoding="utf-8")
        process_corpus(source,folder/f"{name}.bin",tokenizer)
    options={"model":{"vocab_size":258,"context_length":4,"d_model":8,"num_layers":1,"num_heads":2,"d_ff":16,"norm_mode":"pre_norm","ffn_type":"swiglu","rope_theta":10000.0},"data":{"train_data_path":"train.bin","val_data_path":"val.bin","tokenizer_path":"tokenizer.json"},"training":{"batch_size":2,"max_steps":4,"learning_rate":0.01,"min_learning_rate":0.001,"warmup_steps":1,"weight_decay":0.01,"grad_clip":1.0,"seed":42,"device":"cpu","eval_interval":2,"save_interval":2},"output":{"output_dir":"outputs"}}
    path=folder/"run.yaml"
    path.write_text(yaml.safe_dump(options),encoding="utf-8")
    return path


def update_config(path,section,key,value):
    options=yaml.safe_load(path.read_text(encoding="utf-8"))
    options[section][key]=value
    path.write_text(yaml.safe_dump(options),encoding="utf-8")


def equal(left,right):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:
            equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right,strict=True):
            equal(a,b)
    else:
        assert left==right


@pytest.mark.parametrize("dtype",["float32","float64"])
def test_cli_train_resume_and_evaluate_match_array_api(entry,project,tmp_path,dtype):
    config=load_config(project)
    training_data=load_token_data(config.data.train_data_path)
    validation_data=load_token_data(config.data.val_data_path)
    try:
        reference=train(config,training_data,validation_data,dtype=getattr(torch,dtype),eval_batches=2,save_checkpoints=False)
        expected=evaluate(reference.model,validation_data,batch_size=2,context_length=4,num_batches=3,generator=torch.Generator().manual_seed(7))
    finally:
        training_data._mmap.close()
        validation_data._mmap.close()
    relative=str(project.relative_to(tmp_path))
    common=["train","--config",relative,"--dtype",dtype,"--eval-batches","2"]
    part=run_cli(entry,[*common,"--stop-after","1"],tmp_path)
    assert part.returncode==0,part.stderr
    summary=json.loads(part.stdout)
    assert summary["completed_updates"]==1
    first_metrics=[json.loads(line) for line in part.stderr.splitlines()]
    assert [row["step"] for row in first_metrics]==[1]
    assert first_metrics[0]["validation_loss"] is None
    resumed=run_cli(entry,[*common,"--resume",summary["checkpoints"][-1]],tmp_path)
    assert resumed.returncode==0,resumed.stderr
    summary=json.loads(resumed.stdout)
    assert summary["completed_updates"]==4
    metrics=first_metrics+[json.loads(line) for line in resumed.stderr.splitlines()]
    assert [row["step"] for row in metrics]==[1,2,3,4]
    assert [row["train_loss"] for row in metrics]==[row.train_loss for row in reference.history]
    saved_path=Path(summary["checkpoints"][-1])
    payload=read_checkpoint(saved_path)
    equal(payload["model_state_dict"],reference.model.state_dict())
    equal(payload["optimizer_state_dict"],reference.optimizer.state_dict())
    for name,generator in reference.generators.items():
        equal(payload["rng_state"]["generators"][name],generator.get_state())
    completed=run_cli(entry,[*common,"--resume",str(saved_path)],tmp_path)
    assert completed.returncode==0,completed.stderr
    assert completed.stderr==""
    assert json.loads(completed.stdout)["checkpoints"]==[]
    assert json.loads(completed.stdout)["last_metrics"] is None
    # 验证不依赖训练文件或输出目录，也不要求恢复训练采样器状态。
    config.data.train_data_path.unlink()
    update_config(project,"output","output_dir","unused-output")
    before={path:path.read_bytes() for path in config.output.output_dir.iterdir()}
    args=["evaluate","--config",relative,"--checkpoint",str(saved_path),"--dtype",dtype,"--eval-batches","3","--seed","7"]
    for _ in range(2):
        result=run_cli(entry,args,tmp_path)
        assert result.returncode==0,result.stderr
        assert result.stderr==""
        report=json.loads(result.stdout)
        assert report["loss"]==expected.loss
        assert report["num_tokens"]==24
        assert report["num_batches"]==3
        assert report["seed"]==7
        assert report["completed_updates"]==4
    assert {path:path.read_bytes() for path in config.output.output_dir.iterdir()}==before
    assert not (project.parent/"unused-output").exists()


@pytest.mark.parametrize("command",["train","evaluate"])
def test_help(entry,tmp_path,command):
    result=run_cli(entry,[command,"--help"],tmp_path)
    assert result.returncode==0
    assert "--eval-batches" in result.stdout
    assert "尚未实现" not in result.stdout
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize("command",["train","evaluate"])
@pytest.mark.parametrize("problem",["vocab","fingerprint","hash","short","empty","missing_metadata","missing_tokenizer","device","gelu","seed","config"])
def test_invalid_inputs_fail_before_output(project,command,problem,capsys):
    folder=project.parent
    if problem=="vocab":
        update_config(project,"model","vocab_size",259)
    elif problem=="fingerprint":
        (folder/"tokenizer.json").unlink()
        BPETokenizer({**{i:bytes([i]) for i in range(256)},256:b"ac",257:b"<E>"},[(b"a",b"c")],["<E>"]).save(folder/"tokenizer.json")
    elif problem=="hash":
        with (folder/"val.bin").open("r+b") as stream:
            stream.write(b"\x00\x00")
    elif problem in ("short","empty"):
        (folder/"val.bin").unlink()
        (folder/"val.bin.json").unlink()
        (folder/"val.txt").write_text("a" if problem=="short" else "",encoding="utf-8")
        process_corpus(folder/"val.txt",folder/"val.bin",BPETokenizer.from_file(folder/"tokenizer.json"))
    elif problem=="missing_metadata":
        (folder/"val.bin.json").unlink()
    elif problem=="missing_tokenizer":
        (folder/"tokenizer.json").unlink()
    elif problem=="device":
        update_config(project,"training","device","mps")
    elif problem=="gelu":
        update_config(project,"model","ffn_type","gelu")
    elif problem=="seed":
        update_config(project,"training","seed",2**64)
    else:
        project.write_text("invalid: true",encoding="utf-8")
    args=[command,"--config",str(project)]
    if command=="evaluate":
        args.extend(["--checkpoint",str(folder/"missing.pt")])
    assert main(args)==2
    captured=capsys.readouterr()
    assert captured.out==""
    assert captured.err.startswith(command+":")
    assert "Traceback" not in captured.err
    assert not (folder/"outputs").exists()


@pytest.mark.parametrize("command,extra",[("train",["--eval-batches","0"]),("train",["--stop-after","-1"]),("train",["--stop-after","5"]),("evaluate",["--eval-batches","-1"]),("evaluate",["--seed","-1"]),("evaluate",["--seed",str(2**64)])])
def test_invalid_options(project,command,extra,capsys):
    args=[command,"--config",str(project),*extra]
    if command=="evaluate":
        args.extend(["--checkpoint","missing.pt"])
    assert main(args)==2
    assert capsys.readouterr().out==""
    assert not (project.parent/"outputs").exists()


@pytest.mark.parametrize("command",["train","evaluate"])
@pytest.mark.parametrize("contents",[b"",b"broken checkpoint"])
def test_corrupt_checkpoint_returns_error(project,command,contents,capsys):
    path=project.parent/"bad.pt"
    path.write_bytes(contents)
    option="--resume" if command=="train" else "--checkpoint"
    assert main([command,"--config",str(project),option,str(path)])==2
    result=capsys.readouterr()
    assert result.out==""
    assert result.err.startswith(command+":")
    assert "Traceback" not in result.err
    assert not (project.parent/"outputs").exists()


@pytest.mark.parametrize("change",["dtype","model","config"])
def test_resume_rejects_mismatches_without_updates(project,change,capsys):
    assert main(["train","--config",str(project),"--stop-after","1"])==0
    capsys.readouterr()
    path=project.parent/"outputs"/"step_00000001.pt"
    before=path.read_bytes()
    extra=[]
    if change=="dtype":
        extra=["--dtype","float64"]
    elif change=="model":
        update_config(project,"model","norm_mode","post_norm")
    else:
        update_config(project,"training","learning_rate",0.02)
    assert main(["train","--config",str(project),"--resume",str(path),*extra])==2
    assert capsys.readouterr().out==""
    assert path.read_bytes()==before
    assert list(path.parent.iterdir())==[path]


def test_memmaps_close_when_training_or_loading_fails(project,monkeypatch,capsys):
    data_module=importlib.import_module("tiny_lm.data")
    original=data_module.load_token_data
    opened=[]
    def tracked(*args,**kwargs):
        result=original(*args,**kwargs)
        opened.append(result)
        return result
    monkeypatch.setattr(data_module,"load_token_data",tracked)
    assert main(["train","--config",str(project),"--resume","missing.pt"])==2
    assert len(opened)==2
    assert all(data._mmap.closed for data in opened)
    opened.clear()
    (project.parent/"train.bin").unlink()
    assert main(["train","--config",str(project)])==2
    assert len(opened)==1 and opened[0]._mmap.closed
    assert capsys.readouterr().out==""


def test_train_callback_observes_saved_steps_and_rejects_invalid_callback(project):
    config=load_config(project)
    data=np.arange(128,dtype=np.int64)%config.model.vocab_size
    with pytest.raises(TypeError,match="on_step"):
        train(config,data,data,on_step=1)
    assert not config.output.output_dir.exists()
    seen=[]
    def report(metrics):
        if metrics.step%config.training.save_interval==0:
            assert read_checkpoint(config.output.output_dir/f"step_{metrics.step:08d}.pt")["iteration"]==metrics.step
        seen.append(metrics)
    result=train(config,data,data,on_step=report)
    assert seen==result.history
