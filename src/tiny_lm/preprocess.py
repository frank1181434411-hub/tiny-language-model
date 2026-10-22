"""
从 Assignment 1 迁移的文本编码与批量写盘，使用带元信息的 token 数据格式。
"""

from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from numbers import Integral
import hashlib
import json
import os
import re
import tempfile
from tiny_lm.tokenizer.tokenizer import BPETokenizer
from tiny_lm.tokenizer.serialization import _unique_fields,tokenizer_fingerprint

DATA_FORMAT="tiny-lm-tokens"
DATA_VERSION=1


def _metadata_path(path:Path)->Path:
    return path.with_name(path.name+".json")


def _fingerprint(tokenizer:BPETokenizer)->str:
    if not isinstance(tokenizer,BPETokenizer):
        raise TypeError("tokenizer must be a BPETokenizer")
    merges=sorted(tokenizer.merges,key=tokenizer.merges.__getitem__)
    return tokenizer_fingerprint(tokenizer.vocab,merges,tokenizer.special_tokens)


def _storage_dtype(vocab_size:int)->str:
    if type(vocab_size) is not int or not 256<=vocab_size<=2**32:
        raise ValueError("vocab_size must be an integer between 256 and 2**32")
    return "<u2" if vocab_size<=2**16 else "<u4"


def _sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_file(source:Path,target:Path)->None:
    # 同目录硬链接用于完整发布，目标已存在时不覆盖。
    os.link(source,target)


def process_corpus(input_paths:str|Path|Sequence[str|Path],output_bin:str|Path,tokenizer:BPETokenizer,*,record_mode:str="file",special_mode:str="recognize",eos_token:str|None=None,write_batch_size:int=1_000_000)->Path:
    """
    默认接收单文件，每个文件作为一条完整文本；列表按传入顺序处理。
    line 模式将每行作为独立记录并保留换行；可显式在每条记录末尾追加 EOS。
    返回数据文件的绝对路径，旁置元信息为原文件名后追加 .json。
    """
    fingerprint=_fingerprint(tokenizer)
    tokenizer._check_special_mode(special_mode)
    if not isinstance(record_mode,str) or record_mode not in ("file","line"):
        raise ValueError("record_mode must be 'file' or 'line'")
    if type(write_batch_size) is not int or write_batch_size<=0:
        raise ValueError("write_batch_size must be a positive integer")
    if eos_token is not None and (not isinstance(eos_token,str) or eos_token not in tokenizer.special_to_id):
        raise ValueError("eos_token must be a registered special token or None")
    if isinstance(input_paths,(str,Path)):
        input_paths=[input_paths]
    elif not isinstance(input_paths,Sequence) or isinstance(input_paths,(bytes,bytearray)):
        raise TypeError("input_paths must be a path or a sequence of paths")
    if not input_paths:
        raise ValueError("input_paths must contain at least one file")
    paths=[]
    for path in input_paths:
        if not isinstance(path,(str,Path)):
            raise TypeError("each input path must be a string or Path")
        path=Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"input path must be a regular file: {path}")
        paths.append(path)
    output_bin=Path(output_bin)
    metadata_path=_metadata_path(output_bin)
    for path in (output_bin,metadata_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"output already exists: {path}")
    output_bin=output_bin.resolve()
    metadata_path=_metadata_path(output_bin)
    if not output_bin.parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {output_bin.parent}")
    import numpy as np
    dtype=_storage_dtype(len(tokenizer.vocab))
    eos_id=None if eos_token is None else tokenizer.special_to_id[eos_token]
    total_tokens=0
    total_records=0
    sources=[]
    digest=hashlib.sha256()
    temporary=[]
    published=[]
    try:
        with tempfile.NamedTemporaryFile(mode="wb",dir=output_bin.parent,prefix=".tiny-lm-",suffix=".tmp",delete=False) as f_out:
            data_temp=Path(f_out.name)
            temporary.append(data_temp)
            for path in paths:
                source_digest=hashlib.sha256()
                source_bytes=0
                source_tokens=0
                source_records=0
                with path.open("r",encoding="utf-8",newline="") as f:
                    records=iter([f.read()]) if record_mode=="file" else iter(f)
                    for text in records:
                        raw=text.encode("utf-8")
                        source_digest.update(raw)
                        source_bytes+=len(raw)
                        ids=tokenizer.encode(text,special_mode=special_mode)
                        if eos_id is not None:
                            ids.append(eos_id)
                        for start in range(0,len(ids),write_batch_size):
                            batch=ids[start:start+write_batch_size]
                            if any(isinstance(i,bool) or not isinstance(i,Integral) or not 0<=i<len(tokenizer.vocab) for i in batch):
                                raise ValueError("encoded token IDs must be integers within the tokenizer vocabulary")
                            data=np.asarray(batch,dtype=dtype).tobytes()
                            f_out.write(data)
                            digest.update(data)
                        source_tokens+=len(ids)
                        source_records+=1
                sources.append({"path":str(path),"sha256":source_digest.hexdigest(),"num_bytes":source_bytes,"num_records":source_records,"num_tokens":source_tokens})
                total_tokens+=source_tokens
                total_records+=source_records
            f_out.flush()
            os.fsync(f_out.fileno())
        metadata={
            "format":DATA_FORMAT,
            "version":DATA_VERSION,
            "dtype":dtype,
            "num_tokens":total_tokens,
            "num_bytes":total_tokens*np.dtype(dtype).itemsize,
            "sha256":digest.hexdigest(),
            "vocab_size":len(tokenizer.vocab),
            "tokenizer_sha256":fingerprint,
            "options":{"record_mode":record_mode,"special_mode":special_mode,"eos_token":eos_token,"eos_id":eos_id},
            "num_records":total_records,
            "sources":sources,
        }
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",newline="\n",dir=output_bin.parent,prefix=".tiny-lm-",suffix=".tmp",delete=False) as f_meta:
            metadata_temp=Path(f_meta.name)
            temporary.append(metadata_temp)
            json.dump(metadata,f_meta,ensure_ascii=True,indent=2)
            f_meta.write("\n")
            f_meta.flush()
            os.fsync(f_meta.fileno())
        # 元信息最后发布，作为两文件完成标记；不宣称跨文件原子事务。
        _publish_file(data_temp,output_bin)
        published.append(output_bin)
        _publish_file(metadata_temp,metadata_path)
        published.append(metadata_path)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return output_bin


def _nonnegative(value:object,name:str)->None:
    if type(value) is not int or value<0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _check_hash(value:object,name:str)->None:
    if not isinstance(value,str) or re.fullmatch("[0-9a-f]{64}",value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _read_metadata(path:Path)->dict:
    with _metadata_path(path).open("r",encoding="utf-8") as f:
        metadata=json.load(f,object_pairs_hook=_unique_fields)
    fields={"format","version","dtype","num_tokens","num_bytes","sha256","vocab_size","tokenizer_sha256","options","num_records","sources"}
    if not isinstance(metadata,dict) or set(metadata)!=fields:
        raise ValueError("metadata must contain exactly the token data v1 fields")
    if metadata["format"]!=DATA_FORMAT or type(metadata["version"]) is not int or metadata["version"]!=DATA_VERSION:
        raise ValueError("unsupported token data format or version")
    if metadata["dtype"]!=_storage_dtype(metadata["vocab_size"]):
        raise ValueError("metadata dtype must match the vocabulary size and use little-endian uint16 or uint32")
    for key in ("num_tokens","num_bytes","num_records"):
        _nonnegative(metadata[key],key)
    itemsize=2 if metadata["dtype"]=="<u2" else 4
    if metadata["num_bytes"]!=metadata["num_tokens"]*itemsize:
        raise ValueError("metadata byte count must match the token count and dtype")
    for key in ("sha256","tokenizer_sha256"):
        _check_hash(metadata[key],key)
    options=metadata["options"]
    if not isinstance(options,dict) or set(options)!={"record_mode","special_mode","eos_token","eos_id"}:
        raise ValueError("metadata options must contain exactly record_mode, special_mode, eos_token and eos_id")
    if options["record_mode"] not in ("file","line") or options["special_mode"] not in ("recognize","ordinary"):
        raise ValueError("unsupported record_mode or special_mode in metadata")
    if options["eos_token"] is None:
        if options["eos_id"] is not None:
            raise ValueError("eos_id must be null when eos_token is null")
    elif not isinstance(options["eos_token"],str) or not options["eos_token"] or type(options["eos_id"]) is not int or not 256<=options["eos_id"]<metadata["vocab_size"]:
        raise ValueError("metadata EOS must have nonempty text and a valid special token ID")
    sources=metadata["sources"]
    if not isinstance(sources,list) or not sources:
        raise ValueError("metadata sources must be a nonempty list")
    for source in sources:
        if not isinstance(source,dict) or set(source)!={"path","sha256","num_bytes","num_records","num_tokens"}:
            raise ValueError("invalid source metadata fields")
        if not isinstance(source["path"],str) or not source["path"]:
            raise ValueError("source path must be a nonempty string")
        _check_hash(source["sha256"],"source sha256")
        for key in ("num_bytes","num_records","num_tokens"):
            _nonnegative(source[key],f"source {key}")
        if options["record_mode"]=="file" and source["num_records"]!=1:
            raise ValueError("file mode must record exactly one record per source")
        if options["eos_token"] is not None and source["num_tokens"]<source["num_records"]:
            raise ValueError("token count must include one EOS per record")
    if sum(source["num_tokens"] for source in sources)!=metadata["num_tokens"] or sum(source["num_records"] for source in sources)!=metadata["num_records"]:
        raise ValueError("source counts must match the total token and record counts")
    return metadata
