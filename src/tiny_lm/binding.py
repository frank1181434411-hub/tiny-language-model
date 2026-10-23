"""
训练输入的内容指纹与 checkpoint 运行信息校验。
数据按逻辑 token 顺序规范化为小端 uint64，与存储 dtype 和数组步幅无关。
"""

import hashlib
import re

DATA_FINGERPRINT_FORMAT="token-ids-u64le-v1"


def _check_hash(value,name):
    if not isinstance(value,str) or re.fullmatch("[0-9a-f]{64}",value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def token_data_fingerprint(data)->dict:
    """
    对实际 token 数组分块计算内容哈希，不信任路径或旁置 JSON 的声明。
    """
    import numpy as np
    if not isinstance(data,np.ndarray) or isinstance(data,np.ma.MaskedArray) or data.ndim!=1 or data.dtype.kind not in ("i","u"):
        raise ValueError("token data must be a one-dimensional unmasked integer NumPy array")
    digest=hashlib.sha256((DATA_FINGERPRINT_FORMAT+"\0").encode("ascii"))
    for start in range(0,len(data),1_000_000):
        chunk=data[start:start+1_000_000]
        if chunk.dtype.kind=="i" and int(chunk.min())<0:
            raise ValueError("token data must contain nonnegative IDs")
        digest.update(np.asarray(chunk,dtype="<u8").tobytes(order="C"))
    return {"sha256":digest.hexdigest(),"num_tokens":len(data)}


def get_tokenizer_fingerprint(tokenizer)->str:
    from tiny_lm.tokenizer.tokenizer import BPETokenizer
    from tiny_lm.tokenizer.serialization import tokenizer_fingerprint
    if not isinstance(tokenizer,BPETokenizer):
        raise TypeError("tokenizer must be a BPETokenizer")
    merges=sorted(tokenizer.merges,key=tokenizer.merges.__getitem__)
    return tokenizer_fingerprint(tokenizer.vocab,merges,tokenizer.special_tokens)


def validate_run_state(state):
    if state is None:
        return
    fields={"data_fingerprint_format","train_data","validation_data","tokenizer_sha256","eval_batches","dtype"}
    if not isinstance(state,dict) or set(state)!=fields:
        raise ValueError("run_state must contain exactly the checkpoint v2 binding fields")
    if state["data_fingerprint_format"]!=DATA_FINGERPRINT_FORMAT:
        raise ValueError("unsupported token data fingerprint format")
    for name in ("train_data","validation_data"):
        data=state[name]
        if not isinstance(data,dict) or set(data)!={"sha256","num_tokens"}:
            raise ValueError(f"run_state.{name} must contain sha256 and num_tokens")
        _check_hash(data["sha256"],name+".sha256")
        if type(data["num_tokens"]) is not int or data["num_tokens"]<0:
            raise ValueError(f"run_state.{name}.num_tokens must be a nonnegative integer")
    if state["tokenizer_sha256"] is not None:
        _check_hash(state["tokenizer_sha256"],"tokenizer_sha256")
    if type(state["eval_batches"]) is not int or state["eval_batches"]<=0:
        raise ValueError("run_state.eval_batches must be a positive integer")
    if state["dtype"] not in ("float16","bfloat16","float32","float64"):
        raise ValueError("run_state.dtype must be float16, bfloat16, float32 or float64")


def build_run_state(train_data,validation_data,*,dtype,eval_batches,tokenizer=None)->dict:
    state={"data_fingerprint_format":DATA_FINGERPRINT_FORMAT,"train_data":token_data_fingerprint(train_data),"validation_data":token_data_fingerprint(validation_data),"tokenizer_sha256":None if tokenizer is None else get_tokenizer_fingerprint(tokenizer),"eval_batches":eval_batches,"dtype":str(dtype).removeprefix("torch.")}
    validate_run_state(state)
    return state


def check_bindings(payload,*,run_state=None,tokenizer_sha256=None,allow_unbound_checkpoint=False):
    """
    仅允许显式跳过历史缺失信息；已有指纹或运行参数不匹配时始终拒绝。
    未请求核对的低层权重加载保留兼容性，不能据此声称输入已验证。
    """
    if type(allow_unbound_checkpoint) is not bool:
        raise TypeError("allow_unbound_checkpoint must be a bool")
    validate_run_state(run_state)
    if run_state is not None and tokenizer_sha256 is not None and tokenizer_sha256!=run_state["tokenizer_sha256"]:
        raise ValueError("requested tokenizer fingerprint conflicts with run_state")
    saved=payload.get("run_state")
    validate_run_state(saved)
    if run_state is not None:
        if saved is None:
            if not allow_unbound_checkpoint:
                raise ValueError("checkpoint has no historical run_state; use allow_unbound_checkpoint only after verifying the original inputs")
        else:
            for key in ("data_fingerprint_format","train_data","validation_data","eval_batches","dtype"):
                if saved[key]!=run_state[key]:
                    raise ValueError(f"checkpoint {key} does not match the current training run")
            if saved["tokenizer_sha256"] is not None and run_state["tokenizer_sha256"] is None:
                raise ValueError("checkpoint is bound to a tokenizer; provide the original tokenizer to resume")
        tokenizer_sha256=run_state["tokenizer_sha256"]
    if tokenizer_sha256 is not None:
        _check_hash(tokenizer_sha256,"tokenizer_sha256")
        recorded=None if saved is None else saved["tokenizer_sha256"]
        if recorded is None:
            if not allow_unbound_checkpoint:
                raise ValueError("checkpoint has no tokenizer fingerprint; use allow_unbound_checkpoint only after verifying the original tokenizer")
        elif recorded!=tokenizer_sha256:
            raise ValueError("checkpoint tokenizer fingerprint does not match the current tokenizer")
