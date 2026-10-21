"""
从 Assignment 1 迁移的随机窗口采样与下一 token 目标构造。
文件加载、数据格式与预处理仍待迁移。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


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
