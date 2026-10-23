"""
内容指纹、历史输入替换、v1/v2 兼容和缺失信息显式放行的回归测试。
"""

import copy
import hashlib
import importlib
import io
import json
import random
import struct
from dataclasses import replace
import pytest
import yaml
from tiny_lm.binding import token_data_fingerprint,build_run_state,check_bindings,validate_run_state,get_tokenizer_fingerprint
from tiny_lm.checkpoint import save_checkpoint,load_checkpoint,read_checkpoint
from tiny_lm.cli import main
from tiny_lm.config import ModelConfig,TrainingConfig,DataConfig,OutputConfig,ProjectConfig
from tiny_lm.data import load_token_data
from tiny_lm.generate import load_generation_model
from tiny_lm.preprocess import process_corpus
from tiny_lm.tokenizer.tokenizer import BPETokenizer
from tiny_lm.train import train

torch=pytest.importorskip("torch")
np=pytest.importorskip("numpy")
pytest.importorskip("regex")
TransformerLM=importlib.import_module("tiny_lm.model.transformer").TransformerLM
AdamW=importlib.import_module("tiny_lm.optim").AdamW


@pytest.fixture(autouse=True)
def preserve_rng():
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    yield
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def tokenizer(alternative=False):
    pair=(b"a",b"c") if alternative else (b"a",b"b")
    return BPETokenizer({**{i:bytes([i]) for i in range(256)},256:pair[0]+pair[1]},[pair],[])


@pytest.fixture
def project(tmp_path):
    tok=tokenizer()
    tok.save(tmp_path/"tokenizer.json")
    model=ModelConfig(vocab_size=257,context_length=3,d_model=8,num_layers=1,num_heads=2,d_ff=16,norm_mode="pre_norm",ffn_type="swiglu",rope_theta=10000.0)
    training=TrainingConfig(batch_size=2,max_steps=2,learning_rate=0.01,min_learning_rate=0.001,warmup_steps=0,weight_decay=0.01,grad_clip=1.0,seed=42,device="cpu",eval_interval=1,save_interval=1)
    config=ProjectConfig(model=model,training=training,data=DataConfig(train_data_path=tmp_path/"train.bin",val_data_path=tmp_path/"val.bin",tokenizer_path=tmp_path/"tokenizer.json"),output=OutputConfig(output_dir=tmp_path/"outputs"))
    for name in ("train","val"):
        (tmp_path/f"{name}.txt").write_text("ab ac abc "*8,encoding="utf-8")
        process_corpus(tmp_path/f"{name}.txt",tmp_path/f"{name}.bin",tok)
    train_data=load_token_data(config.data.train_data_path)
    val_data=load_token_data(config.data.val_data_path)
    try:
        result=train(config,train_data,val_data,tokenizer=tok,eval_batches=1,stop_after=1)
    finally:
        train_data._mmap.close()
        val_data._mmap.close()
    path=tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()),encoding="utf-8")
    return config,path,result.checkpoints[-1]


@pytest.mark.parametrize("dtype",["<u2",">u2","<u4",">i8","<i8"])
def test_logical_hash_matches_canonical_bytes(dtype):
    data=np.array([0,1,255,256,65535],dtype=dtype)
    expected=hashlib.sha256(b"token-ids-u64le-v1\0"+struct.pack("<5Q",0,1,255,256,65535)).hexdigest()
    assert token_data_fingerprint(data)=={"sha256":expected,"num_tokens":5}
    strided=np.repeat(data,2)[::2]
    assert token_data_fingerprint(strided)==token_data_fingerprint(data)
    assert token_data_fingerprint(data[::-1])!=token_data_fingerprint(data)


def test_hash_chunk_boundary_and_content():
    data=np.zeros(1_000_002,dtype=np.uint16)
    before=token_data_fingerprint(data)
    data[-1]=1
    assert token_data_fingerprint(data)!=before
    assert token_data_fingerprint(data[:-1])!=before


@pytest.mark.parametrize("data",[[1],np.ones((2,2),dtype=int),np.array([1.]),np.array([-1]),np.ma.array([1])])
def test_invalid_array_fingerprints(data):
    with pytest.raises(ValueError):
        token_data_fingerprint(data)


@pytest.mark.parametrize("field,value",[("eval_batches",0),("eval_batches",True),("dtype","bad"),("tokenizer_sha256","A"*64),("data_fingerprint_format","other"),("train_data",{"sha256":"0"*64,"num_tokens":True}),("validation_data",{"sha256":"bad","num_tokens":3})])
def test_malformed_run_state_rejected(field,value):
    state=build_run_state(np.arange(8),np.arange(8),dtype="float32",eval_batches=1)
    state[field]=value
    with pytest.raises(ValueError):
        validate_run_state(state)


@pytest.mark.parametrize("field",["train_data","validation_data","eval_batches","dtype","tokenizer_sha256"])
@pytest.mark.parametrize("allow",[False,True])
def test_recorded_mismatch_never_bypassed(field,allow):
    state=build_run_state(np.arange(8),np.arange(8),dtype="float32",eval_batches=1,tokenizer=tokenizer())
    expected=copy.deepcopy(state)
    if field in ("train_data","validation_data"):
        expected[field]=token_data_fingerprint(np.arange(8)[::-1])
    elif field=="eval_batches":
        expected[field]=2
    elif field=="dtype":
        expected[field]="float64"
    else:
        expected[field]=get_tokenizer_fingerprint(tokenizer(True))
    with pytest.raises(ValueError,match="match"):
        check_bindings({"run_state":state},run_state=expected,allow_unbound_checkpoint=allow)


def rewrite_data(config,name,tok,text):
    folder=config.data.tokenizer_path.parent
    (folder/f"{name}.bin").unlink()
    (folder/f"{name}.bin.json").unlink()
    (folder/f"{name}.txt").write_text(text,encoding="utf-8")
    process_corpus(folder/f"{name}.txt",folder/f"{name}.bin",tok)


@pytest.mark.parametrize("name",["train","val"])
def test_cli_rejects_same_path_replaced_data_and_metadata(project,name,capsys):
    config,path,checkpoint=project
    before=checkpoint.read_bytes()
    rewrite_data(config,name,tokenizer(),"different input "*10)
    assert main(["train","--config",str(path),"--resume",str(checkpoint),"--eval-batches","1","--allow-unbound-checkpoint"])==2
    result=capsys.readouterr()
    assert result.out==""
    assert ("train_data" if name=="train" else "validation_data") in result.err
    assert checkpoint.read_bytes()==before
    assert not (checkpoint.parent/"step_00000002.pt").exists()


@pytest.mark.parametrize("command",["train","evaluate","generate"])
def test_same_size_tokenizer_swap_rejected_in_every_command(project,command,capsys):
    config,path,checkpoint=project
    config.data.tokenizer_path.unlink()
    tok=tokenizer(True)
    tok.save(config.data.tokenizer_path)
    # 保持词表大小相同，替换合并规则，检查已记录 tokenizer 的绑定。
    if command=="train":
        # 保留原数组，用新 tokenizer 的合法元信息重建相同内容。
        for name in ("train","val"):
            # 直接更新当前数据声明，文件加载可通过；历史绑定仍必须拒绝。
            meta=config.data.tokenizer_path.parent/f"{name}.bin.json"
            value=json.loads(meta.read_text(encoding="utf-8"))
            value["tokenizer_sha256"]=get_tokenizer_fingerprint(tok)
            meta.write_text(json.dumps(value),encoding="utf-8")
    elif command=="evaluate":
        rewrite_data(config,"val",tok,"a"*100)
    args=[command,"--config",str(path),"--allow-unbound-checkpoint"]
    if command=="train":
        args.extend(["--resume",str(checkpoint),"--eval-batches","1"])
    else:
        args.extend(["--checkpoint",str(checkpoint)])
    if command=="generate":
        args.extend(["--prompt","a","--max-new-tokens","0"])
    assert main(args)==2
    assert "tokenizer fingerprint" in capsys.readouterr().err


def test_v2_run_state_and_eval_batch_mismatch(project,capsys):
    config,path,checkpoint=project
    saved=read_checkpoint(checkpoint)
    assert saved["format_version"]==2
    assert saved["run_state"]["eval_batches"]==1
    assert saved["run_state"]["dtype"]=="float32"
    assert saved["run_state"]["tokenizer_sha256"]==get_tokenizer_fingerprint(tokenizer())
    assert main(["train","--config",str(path),"--resume",str(checkpoint),"--eval-batches","2"])==2
    assert "eval_batches" in capsys.readouterr().err


@pytest.mark.parametrize("version",[1,2])
@pytest.mark.parametrize("command",["train","evaluate","generate"])
def test_unbound_requires_opt_in_and_resume_upgrades(project,version,command,capsys):
    config,path,checkpoint=project
    saved=read_checkpoint(checkpoint)
    saved["format_version"]=version
    if version==1:
        del saved["run_state"]
    else:
        saved["run_state"]=None
    torch.save(saved,checkpoint)
    assert read_checkpoint(checkpoint)["format_version"]==version
    args=[command,"--config",str(path)]
    if command=="train":
        args.extend(["--resume",str(checkpoint),"--eval-batches","1"])
    else:
        args.extend(["--checkpoint",str(checkpoint)])
    if command=="generate":
        args.extend(["--prompt","a","--max-new-tokens","0"])
    assert main(args)==2
    rejected=capsys.readouterr()
    assert rejected.out==""
    assert ("no historical run_state" if command=="train" else "no tokenizer fingerprint") in rejected.err
    assert main([*args,"--allow-unbound-checkpoint"])==0
    result=capsys.readouterr()
    assert "Warning:" in result.err
    if command=="train":
        upgraded=read_checkpoint(config.output.output_dir/"step_00000002.pt")
        assert upgraded["format_version"]==2
        assert upgraded["run_state"]["tokenizer_sha256"]==get_tokenizer_fingerprint(tokenizer())


def test_mismatch_load_does_not_modify_weights_optimizer_or_rng(project):
    config,_,path=project
    model=TransformerLM.from_config(config.model)
    optimizer=AdamW(model.parameters())
    before={key:value.clone() for key,value in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    expected=copy.deepcopy(read_checkpoint(path)["run_state"])
    expected["eval_batches"]=2
    with pytest.raises(ValueError,match="eval_batches"):
        load_checkpoint(path,model,optimizer,run_state=expected)
    assert torch.equal(rng,torch.get_rng_state())
    assert not optimizer.state
    assert all(torch.equal(value,before[key]) for key,value in model.state_dict().items())


def test_plain_array_training_binds_data_and_requires_recorded_tokenizer(project):
    config,_,path=project
    data=np.arange(32,dtype=np.int64)
    array_config=replace(config,output=OutputConfig(output_dir=config.output.output_dir/"arrays"))
    result=train(array_config,data,data,stop_after=1)
    state=read_checkpoint(result.checkpoints[-1])["run_state"]
    assert state["tokenizer_sha256"] is None
    assert state["train_data"]==token_data_fingerprint(data)
    with pytest.raises(ValueError,match="train_data"):
        train(array_config,data[::-1],data,resume_from=result.checkpoints[-1])
    with pytest.raises(ValueError,match="provide the original tokenizer"):
        check_bindings(read_checkpoint(path),run_state={**read_checkpoint(path)["run_state"],"tokenizer_sha256":None},allow_unbound_checkpoint=True)


def test_bad_binding_schema_rejected_on_read_and_write(project):
    config,_,path=project
    model=TransformerLM.from_config(config.model)
    state=copy.deepcopy(read_checkpoint(path)["run_state"])
    state["extra"]=1
    target=io.BytesIO()
    with pytest.raises(ValueError,match="run_state"):
        save_checkpoint(model,AdamW(model.parameters()),0,target,run_state=state)
    assert target.getvalue()==b""
    payload=read_checkpoint(path)
    payload["run_state"]=state
    torch.save(payload,target)
    target.seek(0)
    with pytest.raises(ValueError,match="run_state"):
        read_checkpoint(target)


def test_generation_loader_checks_tokenizer_before_initialization(project):
    _,_,path=project
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match="tokenizer fingerprint"):
        load_generation_model(path,tokenizer=tokenizer(True))
    assert torch.equal(rng,torch.get_rng_state())


def test_partial_binding_only_allows_missing_tokenizer():
    data=np.arange(8)
    saved=build_run_state(data,data,dtype="float32",eval_batches=1)
    expected=build_run_state(data,data,dtype="float32",eval_batches=1,tokenizer=tokenizer())
    with pytest.raises(ValueError,match="no tokenizer fingerprint"):
        check_bindings({"run_state":saved},run_state=expected)
    check_bindings({"run_state":saved},run_state=expected,allow_unbound_checkpoint=True)
    expected["eval_batches"]=2
    with pytest.raises(ValueError,match="eval_batches"):
        check_bindings({"run_state":saved},run_state=expected,allow_unbound_checkpoint=True)


def test_hash_independent_of_tokenizer_json_format(project):
    config,_,path=project
    text=json.loads(config.data.tokenizer_path.read_text(encoding="utf-8"))
    config.data.tokenizer_path.write_text(json.dumps(text,indent=4,sort_keys=True),encoding="utf-8")
    restored=BPETokenizer.from_file(config.data.tokenizer_path)
    load_generation_model(path,tokenizer=restored)


@pytest.mark.parametrize("version",[True,3])
def test_unknown_checkpoint_versions_rejected(project,version):
    _,_,path=project
    payload=read_checkpoint(path)
    payload["format_version"]=version
    stream=io.BytesIO()
    torch.save(payload,stream)
    stream.seek(0)
    with pytest.raises(ValueError,match="format_version"):
        read_checkpoint(stream)


def test_run_state_dtype_must_match_saved_weights(project):
    config,_,path=project
    state=read_checkpoint(path)["run_state"]
    state["dtype"]="float64"
    model=TransformerLM.from_config(config.model)
    with pytest.raises(ValueError,match="dtype"):
        save_checkpoint(model,AdamW(model.parameters()),0,io.BytesIO(),run_state=state)
