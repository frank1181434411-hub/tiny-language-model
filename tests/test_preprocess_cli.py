"""
通过模块与控制台入口验证预处理命令的真实文件输出及失败保护。
"""

import builtins
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
from tiny_lm.cli import main
from tiny_lm.tokenizer.tokenizer import BPETokenizer


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
def corpus(tmp_path):
    pytest.importorskip("regex")
    pytest.importorskip("numpy")
    (tmp_path/"输入 text.txt").write_bytes("你好 world!\r\n<S>ab ab\n".encode())
    tokenizer=BPETokenizer({**{i:bytes([i]) for i in range(256)},256:b"ab",257:b"<S>",258:b"<E>"},[(b"a",b"b")],["<S>","<E>"])
    tokenizer.save(tmp_path/"tok.json")
    return tokenizer


@pytest.mark.parametrize("command",["train-tokenizer","prepare-data"])
def test_preprocess_help(entry,tmp_path,command):
    result=run_cli(entry,[command,"--help"],tmp_path)
    assert result.returncode==0
    assert "尚未实现" not in result.stdout
    assert ("--special-token" if command=="train-tokenizer" else "--record-mode") in result.stdout
    assert list(tmp_path.iterdir())==[]


def test_train_and_prepare_roundtrip(entry,tmp_path,corpus):
    from tiny_lm.data import load_token_data
    result=run_cli(entry,["train-tokenizer","--input","输入 text.txt","--vocab-size","300","--special-token","<S>","--special-token","<E>","--output","trained.json"],tmp_path)
    assert result.returncode==0,result.stderr
    tokenizer=BPETokenizer.from_file(tmp_path/"trained.json")
    report=json.loads(result.stdout)
    assert report["output"]==str(tmp_path/"trained.json")
    assert report["vocab_size"]==len(tokenizer.vocab)<300
    assert report["num_merges"]==len(tokenizer.merges)
    assert report["special_tokens"]==tokenizer.special_tokens==["<S>","<E>"]
    result=run_cli(entry,["prepare-data","--input","输入 text.txt","--tokenizer","trained.json","--output","tokens.bin"],tmp_path)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report["metadata"]==str(tmp_path/"tokens.bin.json")
    tokens=load_token_data(report["output"],tokenizer=tokenizer)
    try:
        text=(tmp_path/"输入 text.txt").read_bytes().decode("utf-8")
        assert tokens.tolist()==tokenizer.encode(text)
        assert tokenizer.decode(tokens,errors="strict")==text
    finally:
        tokens._mmap.close()


def test_no_implicit_special_tokens(entry,tmp_path,corpus):
    result=run_cli(entry,["train-tokenizer","--input","输入 text.txt","--vocab-size","256","--output","plain.json"],tmp_path)
    assert result.returncode==0,result.stderr
    tokenizer=BPETokenizer.from_file(tmp_path/"plain.json")
    assert tokenizer.special_tokens==[]
    assert len(tokenizer.vocab)==256


@pytest.mark.parametrize("record_mode",["file","line"])
@pytest.mark.parametrize("special_mode",["recognize","ordinary"])
def test_order_records_modes_and_eos(entry,tmp_path,corpus,record_mode,special_mode):
    from tiny_lm.data import load_token_data
    (tmp_path/"second.txt").write_bytes(b"b\na")
    paths=["second.txt","输入 text.txt"]
    args=["prepare-data","--input",paths[0],"--input",paths[1],"--tokenizer","tok.json","--output","tokens.bin","--record-mode",record_mode,"--special-mode",special_mode,"--eos-token","<E>","--write-batch-size","1"]
    result=run_cli(entry,args,tmp_path)
    assert result.returncode==0,result.stderr
    expected=[]
    for name in paths:
        with (tmp_path/name).open(encoding="utf-8",newline="") as f:
            records=[f.read()] if record_mode=="file" else list(f)
        for text in records:
            expected.extend(corpus.encode(text,special_mode=special_mode))
            expected.append(corpus.special_to_id["<E>"])
    tokens=load_token_data(tmp_path/"tokens.bin",tokenizer=corpus)
    try:
        assert tokens.tolist()==expected
    finally:
        tokens._mmap.close()
    metadata=json.loads((tmp_path/"tokens.bin.json").read_text(encoding="utf-8"))
    assert [Path(source["path"]).name for source in metadata["sources"]]==paths
    assert metadata["options"]=={"record_mode":record_mode,"special_mode":special_mode,"eos_token":"<E>","eos_id":258}


@pytest.mark.parametrize("command,extra,message",[("train-tokenizer",["--vocab-size","255"],"vocab_size"),("train-tokenizer",["--special-token","<S>","--special-token","<S>"],"special_tokens"),("train-tokenizer",["--special-token",""],"special_tokens"),("prepare-data",["--eos-token","<unknown>"],"eos_token"),("prepare-data",["--write-batch-size","0"],"write_batch_size"),("prepare-data",["--record-mode","chunk"],"invalid choice"),("prepare-data",["--special-mode","bad"],"invalid choice"),("prepare-data",["--tokenizer","missing.json"],"missing.json")])
def test_invalid_parameters_leave_no_outputs(entry,tmp_path,corpus,command,extra,message):
    args=[command,"--input","输入 text.txt","--output","new.out"]
    args.extend(["--vocab-size","300"] if command=="train-tokenizer" else ["--tokenizer","tok.json"])
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    result=run_cli(entry,[*args,*extra],tmp_path)
    assert result.returncode==2
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout==""
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()}==before


@pytest.mark.parametrize("command,problem",[(command,problem) for command in ("train-tokenizer","prepare-data") for problem in ("missing_input","invalid_utf8","missing_parent","existing_output")]+[("prepare-data","corrupt_tokenizer")])
def test_file_errors_preserve_files(entry,tmp_path,corpus,command,problem):
    if problem=="missing_input":
        (tmp_path/"输入 text.txt").unlink()
    elif problem=="invalid_utf8":
        (tmp_path/"输入 text.txt").write_bytes(b"\xff")
    elif problem=="existing_output":
        (tmp_path/"new.out").write_bytes(b"keep me")
    elif problem=="corrupt_tokenizer":
        (tmp_path/"tok.json").write_text("{",encoding="utf-8")
    output="missing/new.out" if problem=="missing_parent" else "new.out"
    args=[command,"--input","输入 text.txt","--output",output]
    args.extend(["--vocab-size","300"] if command=="train-tokenizer" else ["--tokenizer","tok.json"])
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    result=run_cli(entry,args,tmp_path)
    assert result.returncode==2
    assert "Traceback" not in result.stderr
    assert result.stdout==""
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()}==before


def test_existing_metadata_prevents_data_output(entry,tmp_path,corpus):
    (tmp_path/"tokens.bin.json").write_bytes(b"existing metadata")
    result=run_cli(entry,["prepare-data","--input","输入 text.txt","--tokenizer","tok.json","--output","tokens.bin"],tmp_path)
    assert result.returncode==2
    assert "already exists" in result.stderr
    assert not (tmp_path/"tokens.bin").exists()
    assert (tmp_path/"tokens.bin.json").read_bytes()==b"existing metadata"


@pytest.mark.parametrize("command,dependency",[("train-tokenizer","regex"),("prepare-data","regex"),("prepare-data","numpy")])
def test_missing_dependency_has_install_hint(tmp_path,corpus,monkeypatch,capsys,command,dependency):
    original_import=builtins.__import__
    def blocked_import(name,*args,**kwargs):
        if name==dependency:
            raise ModuleNotFoundError(f"No module named '{dependency}'")
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",blocked_import)
    args=[command,"--input",str(tmp_path/"输入 text.txt"),"--output",str(tmp_path/"out")]
    args.extend(["--vocab-size","300"] if command=="train-tokenizer" else ["--tokenizer",str(tmp_path/"tok.json")])
    assert main(args)==2
    captured=capsys.readouterr()
    assert "pip install" in captured.err
    assert captured.out==""
    assert not (tmp_path/"out").exists()
    assert not list(tmp_path.glob(".tiny-lm-*"))
