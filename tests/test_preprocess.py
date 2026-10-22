from itertools import product
from pathlib import Path
import hashlib
import json
import shutil
import pytest
from tiny_lm.data import load_token_data
from tiny_lm.preprocess import process_corpus,_storage_dtype
from tiny_lm.tokenizer.tokenizer import BPETokenizer
from tiny_lm.tokenizer.serialization import tokenizer_fingerprint
import tiny_lm.preprocess as preprocess

np=pytest.importorskip("numpy")
pytest.importorskip("regex")


def make_tokenizer(merges=(),specials=("<EOS>",)):
    tokens=[bytes([i]) for i in range(256)]+[left+right for left,right in merges]+[token.encode("utf-8") for token in specials]
    return BPETokenizer(dict(enumerate(tokens)),list(merges),list(specials))


def source_file(tmp_path,name="source.txt",text="ab你好🙂\r\n<EOS>"):
    path=tmp_path/name
    path.write_bytes(text.encode("utf-8"))
    return path


def read_metadata(path):
    return json.loads(path.with_name(path.name+".json").read_text(encoding="utf-8"))


def write_metadata(path,metadata):
    path.with_name(path.name+".json").write_text(json.dumps(metadata),encoding="utf-8")


def close_data(data):
    if isinstance(data,np.memmap):
        data._mmap.close()


@pytest.mark.parametrize("batch_size",[1,2,7,1_000_000])
def test_single_file_matches_encoding_and_metadata(tmp_path,batch_size):
    text="ab你好🙂\r\n<EOS>\x00"
    source=source_file(tmp_path,text=text)
    tokenizer=make_tokenizer([(b"a",b"b")])
    output=process_corpus(source,tmp_path/"train.bin",tokenizer,write_batch_size=batch_size)
    assert output==tmp_path/"train.bin"
    data=load_token_data(output,tokenizer=tokenizer)
    try:
        assert isinstance(data,np.memmap)
        assert data.tolist()==tokenizer.encode(text)
        assert not data.flags.writeable
        with pytest.raises(ValueError):
            data[0]=0
    finally:
        close_data(data)
    metadata=read_metadata(output)
    assert metadata["dtype"]=="<u2"
    assert metadata["num_records"]==1
    assert metadata["num_tokens"]==len(tokenizer.encode(text))
    assert metadata["num_bytes"]==output.stat().st_size
    assert metadata["sha256"]==hashlib.sha256(output.read_bytes()).hexdigest()
    assert metadata["sources"]==[{"path":str(source.resolve()),"sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"num_bytes":source.stat().st_size,"num_records":1,"num_tokens":len(tokenizer.encode(text))}]
    assert metadata["options"]=={"record_mode":"file","special_mode":"recognize","eos_token":None,"eos_id":None}
    assert not list(tmp_path.glob(".tiny-lm-*.tmp"))


def test_multiple_files_preserve_order_and_boundaries(tmp_path):
    a=source_file(tmp_path,"a.txt","a")
    b=source_file(tmp_path,"b.txt","b")
    tokenizer=make_tokenizer([(b"a",b"b")])
    output=process_corpus([b,a,b],tmp_path/"multi.bin",tokenizer)
    data=load_token_data(output,tokenizer=tokenizer)
    try:
        assert data.tolist()==[98,97,98]
    finally:
        close_data(data)
    assert tokenizer.encode("bab")==[98,256]
    assert [source["path"] for source in read_metadata(output)["sources"]]==[str(b),str(a),str(b)]
    assert read_metadata(output)["num_records"]==3


@pytest.mark.parametrize("mode,records",[("file",["a\r\nb\n\r\n"]),("line",["a\r\n","b\n","\r\n"])])
def test_record_modes_and_eos(tmp_path,mode,records):
    source=source_file(tmp_path,text="a\r\nb\n\r\n")
    tokenizer=make_tokenizer()
    output=process_corpus(source,tmp_path/"records.bin",tokenizer,record_mode=mode,eos_token="<EOS>")
    expected=[i for record in records for i in tokenizer.encode(record)+[256]]
    data=load_token_data(output,tokenizer=tokenizer)
    try:
        assert data.tolist()==expected
    finally:
        close_data(data)
    metadata=read_metadata(output)
    assert metadata["num_records"]==len(records)
    assert metadata["sources"][0]["sha256"]==hashlib.sha256(source.read_bytes()).hexdigest()


def test_ordinary_mode_and_explicit_eos(tmp_path):
    source=source_file(tmp_path,text="<EOS>")
    tokenizer=make_tokenizer()
    output=process_corpus(source,tmp_path/"ordinary.bin",tokenizer,special_mode="ordinary",eos_token="<EOS>")
    data=load_token_data(output,tokenizer=tokenizer)
    try:
        assert data.tolist()==list(b"<EOS>")+[256]
    finally:
        close_data(data)


@pytest.mark.parametrize("mode,eos,expected,records",[("file",None,[],1),("line",None,[],0),("file","<EOS>",[256],1),("line","<EOS>",[],0)])
def test_empty_input(tmp_path,mode,eos,expected,records):
    source=source_file(tmp_path,text="")
    tokenizer=make_tokenizer()
    output=process_corpus(source,tmp_path/"empty.bin",tokenizer,record_mode=mode,eos_token=eos)
    data=load_token_data(output,tokenizer=tokenizer)
    try:
        assert data.tolist()==expected
        assert not data.flags.writeable
        assert read_metadata(output)["num_records"]==records
    finally:
        close_data(data)


@pytest.mark.parametrize("size,dtype",[(256,"<u2"),(65536,"<u2"),(65537,"<u4"),(2**32,"<u4")])
def test_storage_boundaries(size,dtype):
    assert _storage_dtype(size)==dtype


@pytest.mark.parametrize("size",[True,255,0,2**32+1,256.0])
def test_invalid_storage_size(size):
    with pytest.raises(ValueError,match="vocab_size"):
        _storage_dtype(size)


def test_real_uint32_output_preserves_large_id(tmp_path):
    merges=list(product([bytes([i]) for i in range(256)],repeat=2))
    tokenizer=make_tokenizer(merges)
    source=source_file(tmp_path,text="a")
    output=process_corpus(source,tmp_path/"large.bin",tokenizer,eos_token="<EOS>")
    eos=tokenizer.special_to_id["<EOS>"]
    assert eos>65535
    assert output.read_bytes()==(97).to_bytes(4,"little")+eos.to_bytes(4,"little")
    data=load_token_data(output,tokenizer=tokenizer)
    try:
        assert data.tolist()==[97,eos]
        assert data.dtype.itemsize==4
    finally:
        close_data(data)


@pytest.mark.parametrize("value",[-1,65536,True,1.5,"1"])
def test_invalid_encoded_ids_do_not_wrap_or_publish(tmp_path,monkeypatch,value):
    source=source_file(tmp_path)
    tokenizer=make_tokenizer()
    monkeypatch.setattr(tokenizer,"encode",lambda *args,**kwargs:[value])
    with pytest.raises(ValueError,match="encoded token IDs"):
        process_corpus(source,tmp_path/"bad.bin",tokenizer)
    assert sorted(path.name for path in tmp_path.iterdir())==[source.name]


@pytest.mark.parametrize("options",[{"record_mode":"chunk"},{"record_mode":None},{"special_mode":"unknown"},{"eos_token":"missing"},{"eos_token":1},{"write_batch_size":0},{"write_batch_size":True},{"write_batch_size":1.5}])
def test_invalid_options_do_not_create_outputs(tmp_path,options):
    source=source_file(tmp_path)
    with pytest.raises((ValueError,TypeError)):
        process_corpus(source,tmp_path/"bad.bin",make_tokenizer(),**options)
    assert sorted(path.name for path in tmp_path.iterdir())==[source.name]


@pytest.mark.parametrize("inputs",[[],None,b"file",1,[None]])
def test_invalid_input_list(tmp_path,inputs):
    with pytest.raises((ValueError,TypeError)):
        process_corpus(inputs,tmp_path/"bad.bin",make_tokenizer())
    assert list(tmp_path.iterdir())==[]


def test_missing_input_and_parent(tmp_path):
    with pytest.raises(FileNotFoundError):
        process_corpus(tmp_path/"missing.txt",tmp_path/"bad.bin",make_tokenizer())
    source=source_file(tmp_path)
    with pytest.raises(FileNotFoundError):
        process_corpus(source,tmp_path/"missing"/"data.bin",make_tokenizer())
    assert not (tmp_path/"missing").exists()


def test_invalid_utf8_later_file_leaves_no_output(tmp_path):
    source=source_file(tmp_path)
    invalid=tmp_path/"invalid.txt"
    invalid.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        process_corpus([source,invalid],tmp_path/"bad.bin",make_tokenizer(),write_batch_size=1)
    assert set(path.name for path in tmp_path.iterdir())=={source.name,invalid.name}


@pytest.mark.parametrize("suffix",["",".json"])
def test_existing_output_is_preserved(tmp_path,suffix):
    source=source_file(tmp_path)
    existing=tmp_path/("data.bin"+suffix)
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        process_corpus(source,tmp_path/"data.bin",make_tokenizer())
    assert existing.read_bytes()==b"keep"
    assert len(list(tmp_path.iterdir()))==2


def test_second_publish_failure_rolls_back_only_new_data(tmp_path,monkeypatch):
    source=source_file(tmp_path)
    real_publish=preprocess._publish_file
    def fail_metadata(source,target):
        if target.suffix==".json":
            raise OSError("injected metadata publish failure")
        real_publish(source,target)
    monkeypatch.setattr(preprocess,"_publish_file",fail_metadata)
    with pytest.raises(OSError,match="injected"):
        process_corpus(source,tmp_path/"data.bin",make_tokenizer())
    assert sorted(path.name for path in tmp_path.iterdir())==[source.name]


def test_publish_race_does_not_overwrite_competing_file(tmp_path,monkeypatch):
    source=source_file(tmp_path)
    real_publish=preprocess._publish_file
    def competing_writer(source,target):
        target.write_bytes(b"other writer")
        real_publish(source,target)
    monkeypatch.setattr(preprocess,"_publish_file",competing_writer)
    output=tmp_path/"data.bin"
    with pytest.raises(FileExistsError):
        process_corpus(source,output,make_tokenizer())
    assert output.read_bytes()==b"other writer"
    assert not output.with_name(output.name+".json").exists()
    assert not list(tmp_path.glob(".tiny-lm-*.tmp"))


@pytest.fixture
def artifact(tmp_path):
    source=source_file(tmp_path,text="abc")
    tokenizer=make_tokenizer()
    output=process_corpus(source,tmp_path/"data.bin",tokenizer)
    return output,tokenizer


@pytest.mark.parametrize("case",["version","bool_version","format","dtype","big_endian","num_tokens","bool_count","num_bytes","hash_format","tokenizer_hash_format","unknown","missing","sources","source_hash","source_count","source_path","options","record_mode","special_mode","eos_id","eos_text"])
def test_bad_metadata_rejected(artifact,case):
    output,tokenizer=artifact
    metadata=read_metadata(output)
    if case in ("version","format","dtype"):
        metadata[case]="unsupported"
    elif case=="bool_version":
        metadata["version"]=True
    elif case=="big_endian":
        metadata["dtype"]=">u2"
    elif case=="num_tokens":
        metadata["num_tokens"]=-1
    elif case=="bool_count":
        metadata["num_tokens"]=True
    elif case=="num_bytes":
        metadata["num_bytes"]+=1
    elif case=="hash_format":
        metadata["sha256"]="bad"
    elif case=="tokenizer_hash_format":
        metadata["tokenizer_sha256"]="BAD"
    elif case=="unknown":
        metadata["unknown"]=0
    elif case=="missing":
        del metadata["version"]
    elif case=="sources":
        metadata["sources"]=[]
    elif case=="source_hash":
        metadata["sources"][0]["sha256"]="bad"
    elif case=="source_count":
        metadata["sources"][0]["num_tokens"]+=1
    elif case=="source_path":
        metadata["sources"][0]["path"]=None
    elif case=="options":
        metadata["options"]["extra"]=1
    elif case in ("record_mode","special_mode"):
        metadata["options"][case]="bad"
    elif case=="eos_id":
        metadata["options"]["eos_id"]=256
    elif case=="eos_text":
        metadata["options"]["eos_token"]="<EOS>"
    write_metadata(output,metadata)
    with pytest.raises(ValueError):
        load_token_data(output,tokenizer=tokenizer)


def test_truncated_and_missing_files(artifact):
    output,tokenizer=artifact
    metadata_path=output.with_name(output.name+".json")
    metadata=metadata_path.read_bytes()
    metadata_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_token_data(output)
    metadata_path.write_bytes(metadata)
    output.write_bytes(output.read_bytes()[:-1])
    with pytest.raises(ValueError,match="file size"):
        load_token_data(output)
    output.unlink()
    with pytest.raises(FileNotFoundError):
        load_token_data(output,tokenizer=tokenizer)


def test_same_size_corruption_and_explicit_hash_skip(artifact):
    output,tokenizer=artifact
    output.write_bytes(np.array([98,98,99],dtype="<u2").tobytes())
    with pytest.raises(ValueError,match="SHA-256"):
        load_token_data(output)
    data=load_token_data(output,tokenizer=tokenizer,verify_hash=False)
    try:
        assert data.tolist()==[98,98,99]
    finally:
        close_data(data)


def test_out_of_range_even_when_hash_matches(artifact):
    output,tokenizer=artifact
    output.write_bytes(np.array([65535,98,99],dtype="<u2").tobytes())
    metadata=read_metadata(output)
    metadata["sha256"]=hashlib.sha256(output.read_bytes()).hexdigest()
    write_metadata(output,metadata)
    with pytest.raises(ValueError,match="outside the vocabulary"):
        load_token_data(output,tokenizer=tokenizer)
    with pytest.raises(ValueError,match="outside the vocabulary"):
        load_token_data(output,verify_hash=False)
    output.unlink()


def test_equal_vocab_size_different_tokenizer_rejected(artifact):
    output,_=artifact
    with pytest.raises(ValueError,match="fingerprint"):
        load_token_data(output,tokenizer=make_tokenizer(specials=["<END>"]))


def test_semantic_fingerprint_ignores_json_layout(tmp_path):
    tokenizer=make_tokenizer([(b"a",b"b")])
    path=tmp_path/"tokenizer.json"
    tokenizer.save(path)
    payload=json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(payload,separators=(",",":")),encoding="utf-8")
    restored=BPETokenizer.from_file(path)
    merges=sorted(tokenizer.merges,key=tokenizer.merges.__getitem__)
    assert tokenizer_fingerprint(tokenizer.vocab,merges,tokenizer.special_tokens)==tokenizer_fingerprint(restored.vocab,merges,restored.special_tokens)
    reversed_vocab=dict(reversed(list(tokenizer.vocab.items())))
    assert tokenizer_fingerprint(reversed_vocab,merges,tokenizer.special_tokens)==tokenizer_fingerprint(tokenizer.vocab,merges,tokenizer.special_tokens)


def test_metadata_eos_must_match_tokenizer(artifact):
    output,tokenizer=artifact
    metadata=read_metadata(output)
    metadata["options"].update(eos_token="<WRONG>",eos_id=256)
    write_metadata(output,metadata)
    with pytest.raises(ValueError,match="EOS"):
        load_token_data(output,tokenizer=tokenizer)


@pytest.mark.parametrize("text",['{"version":1,"version":1}',"{","null","[]"])
def test_invalid_json(artifact,text):
    output,_=artifact
    output.with_name(output.name+".json").write_text(text,encoding="utf-8")
    with pytest.raises(ValueError):
        load_token_data(output)


def test_artifact_can_move_without_source_files(artifact,tmp_path):
    output,tokenizer=artifact
    moved=tmp_path/"moved"
    moved.mkdir()
    for path in (output,output.with_name(output.name+".json")):
        shutil.copyfile(path,moved/path.name)
    (tmp_path/"source.txt").unlink()
    data=load_token_data(moved/output.name,tokenizer=tokenizer)
    try:
        assert data.tolist()==[97,98,99]
    finally:
        close_data(data)


def test_relative_paths_follow_working_directory(tmp_path,monkeypatch):
    source_file(tmp_path,text="abc")
    monkeypatch.chdir(tmp_path)
    output=process_corpus("source.txt","data.bin",make_tokenizer())
    assert output==tmp_path/"data.bin"
    data=load_token_data(Path("data.bin"))
    try:
        assert data.tolist()==[97,98,99]
    finally:
        close_data(data)


def test_loaded_data_works_with_sampler(artifact):
    torch=pytest.importorskip("torch")
    from tiny_lm.data import get_batch
    output,tokenizer=artifact
    data=load_token_data(output,tokenizer=tokenizer)
    try:
        x,y=get_batch(data,1,2,"cpu",generator=torch.Generator().manual_seed(1))
        assert x.tolist()==[[97,98]]
        assert y.tolist()==[[98,99]]
    finally:
        close_data(data)
