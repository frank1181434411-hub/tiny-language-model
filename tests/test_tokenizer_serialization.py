import base64
import json
import pytest
from tiny_lm.tokenizer.serialization import load_tokenizer,save_tokenizer


def example():
    vocab={i:bytes([i]) for i in range(256)}
    vocab.update({256:b"ab",257:b"abc",258:"<结束>".encode(),259:b"a"})
    return vocab,[(b"a",b"b"),(b"ab",b"c")],["<结束>","a"]


def test_round_trip_all_bytes_and_special_overlap(tmp_path):
    path=tmp_path/"tokenizer.json"
    data=example()
    save_tokenizer(path,*data)
    assert load_tokenizer(path)==data
    payload=json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"]==1
    assert payload["special_tokens"]==[{"text":"<结束>","id":258},{"text":"a","id":259}]
    assert payload["vocab"][97]==payload["vocab"][259]
    assert payload["pretokenizer"]=="gpt2-regex-v1"
    second=tmp_path/"second.json"
    save_tokenizer(second,*load_tokenizer(path))
    assert second.read_bytes()==path.read_bytes()


def test_base_only_round_trip(tmp_path):
    data=({i:bytes([i]) for i in range(256)},[],[])
    path=tmp_path/"base.json"
    save_tokenizer(path,*data)
    assert load_tokenizer(path)==data


def test_save_protects_existing_file(tmp_path):
    path=tmp_path/"existing.json"
    path.write_text("keep",encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_tokenizer(path,*example())
    assert path.read_text(encoding="utf-8")=="keep"


def test_save_does_not_create_parent(tmp_path):
    path=tmp_path/"missing"/"tokenizer.json"
    with pytest.raises(FileNotFoundError):
        save_tokenizer(path,*example())
    assert not path.parent.exists()


@pytest.mark.parametrize("case",["id_gap","bool_id","base_bytes","text_value","empty_bytes","merge_shape","merge_type","merge_dependency","merge_output","duplicate_pair","special_bytes","special_duplicate"])
def test_invalid_data_rejected_before_write(tmp_path,case):
    vocab,merges,specials=example()
    if case=="id_gap":
        del vocab[10]
    elif case=="bool_id":
        vocab.pop(1)
        vocab[True]=b"\x01"
    elif case=="base_bytes":
        vocab[10]=b"x"
    elif case=="text_value":
        vocab[256]="ab"
    elif case=="empty_bytes":
        vocab[256]=b""
    elif case=="merge_shape":
        merges[0]=(b"a",)
    elif case=="merge_type":
        merges[0]=("a",b"b")
    elif case=="merge_dependency":
        merges.reverse()
    elif case=="merge_output":
        vocab[256]=b"ac"
    elif case=="duplicate_pair":
        merges[1]=merges[0]
        vocab[257]=vocab[256]
    elif case=="special_bytes":
        vocab[258]=b"wrong"
    elif case=="special_duplicate":
        specials[1]=specials[0]
    path=tmp_path/"invalid.json"
    with pytest.raises((ValueError,TypeError)):
        save_tokenizer(path,vocab,merges,specials)
    assert not path.exists()


@pytest.mark.parametrize("case",["format","version","bool_version","pretokenizer","unknown_field","missing_field","vocab_type","merges_type","specials_type","base64","noncanonical","empty_bytes","base64_type","merge_shape","merge_dependency","merge_output","base_byte","id_gap","special_id","bool_special_id","special_text","special_extra","duplicate_special","special_bytes"])
def test_invalid_artifact(tmp_path,case):
    path=tmp_path/"tokenizer.json"
    save_tokenizer(path,*example())
    payload=json.loads(path.read_text(encoding="utf-8"))
    if case in ("format","version","pretokenizer"):
        payload[case]="unsupported"
    elif case=="bool_version":
        payload["version"]=True
    elif case=="unknown_field":
        payload["extra"]=0
    elif case=="missing_field":
        del payload["format"]
    elif case in ("vocab_type","merges_type","specials_type"):
        key={"vocab_type":"vocab","merges_type":"merges","specials_type":"special_tokens"}[case]
        payload[key]={}
    elif case in ("base64","noncanonical","empty_bytes","base64_type"):
        payload["vocab"][0]={"base64":"?","noncanonical":"AB==","empty_bytes":"","base64_type":1}[case]
    elif case=="merge_shape":
        payload["merges"][0]=["YQ=="]
    elif case=="merge_dependency":
        payload["merges"].reverse()
    elif case=="merge_output":
        payload["vocab"][256]=base64.b64encode(b"bad").decode()
    elif case=="base_byte":
        payload["vocab"][0]="YQ=="
    elif case=="id_gap":
        payload["vocab"].pop()
    elif case in ("special_id","bool_special_id"):
        payload["special_tokens"][0]["id"]=True if case=="bool_special_id" else 0
    elif case=="special_text":
        payload["special_tokens"][0]["text"]=1
    elif case=="special_extra":
        payload["special_tokens"][0]["extra"]=0
    elif case=="duplicate_special":
        payload["special_tokens"][1]["text"]=payload["special_tokens"][0]["text"]
    elif case=="special_bytes":
        payload["special_tokens"][0]["text"]="wrong"
    path.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises((ValueError,TypeError)):
        load_tokenizer(path)


@pytest.mark.parametrize("text",['{"version":1,"version":1}',"[]","null","{invalid"])
def test_malformed_json(tmp_path,text):
    path=tmp_path/"invalid.json"
    path.write_text(text,encoding="utf-8")
    with pytest.raises(ValueError):
        load_tokenizer(path)


def test_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tokenizer(tmp_path/"missing.json")
