"""
从 Assignment 1 迁移的随机窗口采样与下一 token 目标构造。
支持带元信息的 token 文件只读加载；文本预处理位于 preprocess.py。
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch
    from tiny_lm.tokenizer.tokenizer import BPETokenizer


def load_token_data(path:str|Path,*,tokenizer:BPETokenizer|None=None,verify_hash:bool=True)->npt.NDArray:
    """
    校验旁置元信息后只读加载；默认完整扫描数据哈希和 ID 范围。
    可传 tokenizer 校验指纹；空数据返回只读空数组，非空返回 memmap。
    """
    from tiny_lm.preprocess import _read_metadata,_fingerprint,_sha256_file
    import numpy as np
    if type(verify_hash) is not bool:
        raise TypeError("verify_hash must be a bool")
    path=Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("token data path must be a regular file")
    metadata=_read_metadata(path)
    if path.stat().st_size!=metadata["num_bytes"]:
        raise ValueError("token data file size does not match metadata")
    if tokenizer is not None:
        if _fingerprint(tokenizer)!=metadata["tokenizer_sha256"]:
            raise ValueError("tokenizer fingerprint does not match token data metadata")
        options=metadata["options"]
        if options["eos_token"] is not None and tokenizer.special_to_id.get(options["eos_token"])!=options["eos_id"]:
            raise ValueError("metadata EOS does not match the tokenizer")
    if verify_hash and _sha256_file(path)!=metadata["sha256"]:
        raise ValueError("token data SHA-256 does not match metadata")
    if metadata["num_tokens"]==0:
        data=np.empty(0,dtype=metadata["dtype"])
        data.flags.writeable=False
        return data
    data=np.memmap(path,dtype=metadata["dtype"],mode="r",shape=(metadata["num_tokens"],))
    try:
        for start in range(0,len(data),1_000_000):
            if int(data[start:start+1_000_000].max())>=metadata["vocab_size"]:
                raise ValueError("token data contains IDs outside the vocabulary")
    except BaseException:
        data._mmap.close()
        raise
    return data


def get_batch(dataset:npt.NDArray,batch_size:int,context_length:int,device:str|torch.device,*,generator:torch.Generator|None=None)->tuple[torch.Tensor,torch.Tensor]:
    """
    从一维整数数组或 memmap 中均匀、有放回地采样，返回指定设备上的 int64 张量。
    generator 须为 CPU generator；省略时使用 PyTorch 全局 CPU 随机状态。
    只检查抽中的 token 是否在 int64 非负范围内，不检查词表上界。
    """
    import numpy as np
    import torch

    if not isinstance(dataset,np.ndarray) or isinstance(dataset,np.ma.MaskedArray):
        raise TypeError("dataset must be a NumPy ndarray or memmap without masked values")
    if dataset.ndim!=1:
        raise ValueError("dataset must be one-dimensional")
    if dataset.dtype.kind not in ("i","u"):
        raise TypeError("dataset must have an integer dtype")
    for name,value in (("batch_size",batch_size),("context_length",context_length)):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value<=0:
            raise ValueError(f"{name} must be positive")
    n=len(dataset)
    if n<=context_length:
        raise ValueError("dataset must contain at least context_length + 1 tokens")
    if generator is not None:
        if not isinstance(generator,torch.Generator):
            raise TypeError("generator must be a CPU torch.Generator or None")
        if generator.device.type!="cpu":
            raise ValueError("generator must be on CPU")

    max_idx=n-context_length-1
    ix=torch.randint(0,max_idx+1,(batch_size,),device="cpu",generator=generator).tolist()
    x_stack=[dataset[i:i+context_length] for i in ix]
    y_stack=[dataset[i+1:i+context_length+1] for i in ix]
    x_array=np.array(x_stack)
    y_array=np.array(y_stack)
    # 转换前检查，避免 uint64 越界后绕回负数；不扫描整个数据集。
    for values in (x_array,y_array):
        if int(values.min())<0 or int(values.max())>np.iinfo(np.int64).max:
            raise ValueError("sampled token IDs must be between 0 and int64 max")
    x=torch.from_numpy(x_array.astype(np.int64,copy=False)).to(device)
    y=torch.from_numpy(y_array.astype(np.int64,copy=False)).to(device)
    return x,y
