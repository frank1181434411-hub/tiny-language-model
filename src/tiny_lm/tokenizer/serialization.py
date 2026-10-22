"""
字节级 BPE 的版本化 JSON 保存与加载。
"""

from pathlib import Path
import base64
import binascii
import json
import hashlib
from tiny_lm.tokenizer.rules import PRETOKENIZER,validate_special_tokens

FORMAT_VERSION=1


def _validate_tokenizer(vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str])->None:
    validate_special_tokens(special_tokens)
    if not isinstance(vocab,dict):
        raise TypeError("vocab must be a dictionary")
    if not isinstance(merges,list):
        raise TypeError("merges must be a list")
    size=256+len(merges)+len(special_tokens)
    if any(type(i) is not int for i in vocab) or set(vocab)!=set(range(size)):
        raise ValueError("vocab IDs must be consecutive integers matching the base bytes, merges and special tokens")
    if any(not isinstance(token,bytes) or not token for token in vocab.values()):
        raise ValueError("vocab values must be nonempty bytes")
    if any(vocab[i]!=bytes([i]) for i in range(256)):
        raise ValueError("vocab IDs 0 through 255 must contain their corresponding single bytes")
    available={bytes([i]) for i in range(256)}
    seen=set()
    for i,pair in enumerate(merges):
        if not isinstance(pair,tuple) or len(pair)!=2 or any(not isinstance(token,bytes) for token in pair):
            raise ValueError("each merge must be a tuple of two byte strings")
        if pair in seen:
            raise ValueError("merges must not contain duplicate pairs")
        if any(token not in available for token in pair):
            raise ValueError(f"merge {i} refers to a token unavailable at that rank")
        token=pair[0]+pair[1]
        if vocab[256+i]!=token:
            raise ValueError(f"vocab entry {256+i} must equal the corresponding merged bytes")
        available.add(token)
        seen.add(pair)
    for i,token in enumerate(special_tokens,start=256+len(merges)):
        if vocab[i]!=token.encode("utf-8"):
            raise ValueError(f"vocab entry {i} must equal the corresponding special token bytes")


def _tokenizer_payload(vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str])->dict:
    _validate_tokenizer(vocab,merges,special_tokens)
    return {
        "format":"tiny-lm-bpe",
        "version":FORMAT_VERSION,
        "pretokenizer":PRETOKENIZER,
        "vocab":[base64.b64encode(vocab[i]).decode("ascii") for i in range(len(vocab))],
        "merges":[[base64.b64encode(token).decode("ascii") for token in pair] for pair in merges],
        "special_tokens":[{"text":token,"id":i} for i,token in enumerate(special_tokens,start=256+len(merges))],
    }


def tokenizer_fingerprint(vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str])->str:
    """
    对规范化词表、规则和特殊 token 元信息计算 SHA-256，不受 JSON 排版影响。
    """
    payload=_tokenizer_payload(vocab,merges,special_tokens)
    content=json.dumps(payload,ensure_ascii=True,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(content.encode("ascii")).hexdigest()


def save_tokenizer(path:str|Path,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str])->None:
    """
    保存原生 v1 文件；父目录须已存在，拒绝覆盖已有文件。
    字节使用 Base64，特殊 token 的独立 ID 明确记录。
    """
    payload=_tokenizer_payload(vocab,merges,special_tokens)
    content=json.dumps(payload,ensure_ascii=False,indent=2)+"\n"
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        f.write(content)


def _unique_fields(pairs:list[tuple[str,object]])->dict:
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key]=value
    return result


def _decode_bytes(value:object)->bytes:
    if not isinstance(value,str):
        raise ValueError("encoded token bytes must be Base64 strings")
    try:
        token=base64.b64decode(value,validate=True)
    except (ValueError,binascii.Error) as exc:
        raise ValueError("encoded token bytes must be valid Base64") from exc
    if not token or base64.b64encode(token).decode("ascii")!=value:
        raise ValueError("encoded token bytes must be nonempty canonical Base64")
    return token


def load_tokenizer(path:str|Path)->tuple[dict[int,bytes],list[tuple[bytes,bytes]],list[str]]:
    """
    加载并校验原生 v1 文件，返回词表、合并规则和特殊 token。
    不自动识别旧版 vocab.json 与 merges.txt。
    """
    with Path(path).open("r",encoding="utf-8") as f:
        payload=json.load(f,object_pairs_hook=_unique_fields)
    fields={"format","version","pretokenizer","vocab","merges","special_tokens"}
    if not isinstance(payload,dict) or set(payload)!=fields:
        raise ValueError("tokenizer JSON must contain exactly the v1 format fields")
    if payload["format"]!="tiny-lm-bpe":
        raise ValueError("unsupported tokenizer format")
    if type(payload["version"]) is not int or payload["version"]!=FORMAT_VERSION:
        raise ValueError("unsupported tokenizer version")
    if payload["pretokenizer"]!=PRETOKENIZER:
        raise ValueError("unsupported pretokenizer rule")
    if any(not isinstance(payload[key],list) for key in ("vocab","merges","special_tokens")):
        raise ValueError("vocab, merges and special_tokens must be lists")
    vocab={i:_decode_bytes(token) for i,token in enumerate(payload["vocab"])}
    merges=[]
    for pair in payload["merges"]:
        if not isinstance(pair,list) or len(pair)!=2:
            raise ValueError("each encoded merge must be a list of two Base64 strings")
        merges.append((_decode_bytes(pair[0]),_decode_bytes(pair[1])))
    special_tokens=[]
    for i,entry in enumerate(payload["special_tokens"],start=256+len(merges)):
        if not isinstance(entry,dict) or set(entry)!={"text","id"}:
            raise ValueError("each special token must contain exactly text and id")
        if type(entry["id"]) is not int or entry["id"]!=i:
            raise ValueError("special token IDs must follow the merge vocabulary in list order")
        special_tokens.append(entry["text"])
    _validate_tokenizer(vocab,merges,special_tokens)
    return vocab,merges,special_tokens
