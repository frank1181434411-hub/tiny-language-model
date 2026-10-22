"""
保存与恢复模型、优化器、已完成更新次数、配置和随机数状态。
路径保存使用同目录临时文件替换；显式调用时才加载计算依赖。
"""

from __future__ import annotations
import copy
import io
import os
import pickle
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from tiny_lm.config import ModelConfig,ProjectConfig

FORMAT_VERSION=1


def _config_dict(config):
    if config is None:
        return None
    if isinstance(config,ProjectConfig):
        return config.to_dict()
    if isinstance(config,ModelConfig):
        return {"model":asdict(config)}
    raise TypeError("config must be a ModelConfig, ProjectConfig or None")


def _class_name(value):
    return type(value).__module__+"."+type(value).__qualname__


def _model_config(model):
    from tiny_lm.model.transformer import TransformerLM
    return model.get_config() if isinstance(model,TransformerLM) else None


def _check_config(model_config,config):
    if model_config is not None and config is not None:
        if any(model_config.get(key)!=value for key,value in config["model"].items()):
            raise ValueError("config.model does not match the model construction configuration")


def _parameter_names(model,optimizer):
    import torch
    if not isinstance(model,torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer,torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    names={id(p):name for name,p in model.named_parameters()}
    result=[]
    seen=set()
    for group in optimizer.param_groups:
        group_names=[]
        for p in group["params"]:
            if id(p) not in names:
                raise ValueError("optimizer parameters must belong to the supplied model")
            if id(p) in seen:
                raise ValueError("optimizer parameters must not contain duplicates")
            seen.add(id(p))
            group_names.append(names[id(p)])
        result.append(group_names)
    return result


def _generators(generators):
    import torch
    if generators is None:
        return {}
    if not isinstance(generators,dict):
        raise TypeError("generators must be a dictionary of named CPU torch.Generator instances")
    for name,generator in generators.items():
        if not isinstance(name,str) or not name:
            raise ValueError("generator names must be nonempty strings")
        if not isinstance(generator,torch.Generator) or generator.device.type!="cpu":
            raise TypeError("generators must contain CPU torch.Generator instances")
    if len({id(generator) for generator in generators.values()})!=len(generators):
        raise ValueError("each generator name must refer to a different generator")
    return generators


def _rng_state(generators):
    import numpy as np
    import torch
    numpy_state=np.random.get_state()
    return dict(python=random.getstate(),numpy=(numpy_state[0],numpy_state[1].tolist(),int(numpy_state[2]),int(numpy_state[3]),float(numpy_state[4])),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],generators={name:generator.get_state() for name,generator in generators.items()})


def _validate_rng(state,generators):
    import numpy as np
    import torch
    if not isinstance(state,dict) or set(state)!={"python","numpy","torch_cpu","torch_cuda","generators"}:
        raise ValueError("checkpoint RNG state has invalid fields")
    if not isinstance(state["generators"],dict) or set(state["generators"])!=set(generators):
        raise ValueError("checkpoint generator names must match the supplied generators")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
        for value in state["generators"].values():
            torch.Generator(device="cpu").set_state(value)
    except (TypeError,ValueError,RuntimeError,OverflowError) as exc:
        raise ValueError("checkpoint contains invalid RNG state") from exc
    cuda=state["torch_cuda"]
    if not isinstance(cuda,list):
        raise ValueError("checkpoint CUDA RNG state must be a list")
    if cuda:
        if not torch.cuda.is_available() or len(cuda)!=torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count does not match this runtime")
        for index,value in enumerate(cuda):
            try:
                torch.Generator(device=f"cuda:{index}").set_state(value)
            except (TypeError,ValueError,RuntimeError) as exc:
                raise ValueError("checkpoint contains invalid CUDA RNG state") from exc


def _restore_rng(state,generators):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for name,generator in generators.items():
        generator.set_state(state["generators"][name])


def _validate_payload(payload):
    if not isinstance(payload,dict):
        raise ValueError("checkpoint must contain a dictionary")
    fields={"format_version","iteration","model_class","model_config","model_state_dict","optimizer_class","optimizer_parameter_names","optimizer_state_dict","config","rng_state"}
    if set(payload)!=fields:
        raise ValueError("checkpoint fields are invalid; unversioned legacy checkpoints are not supported")
    if type(payload["format_version"]) is not int or payload["format_version"]!=FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format_version")
    if type(payload["iteration"]) is not int or payload["iteration"]<0:
        raise ValueError("iteration must be the nonnegative number of completed optimizer updates")
    for name in ("model_state_dict","optimizer_state_dict","rng_state"):
        if not isinstance(payload[name],dict):
            raise ValueError(f"checkpoint {name} must be a dictionary")
    if payload["model_config"] is not None and not isinstance(payload["model_config"],dict):
        raise ValueError("checkpoint model_config must be a dictionary or None")
    if payload["config"] is not None:
        if not isinstance(payload["config"],dict) or not isinstance(payload["config"].get("model"),dict):
            raise ValueError("checkpoint config must contain a model dictionary")
        _check_config(payload["model_config"],payload["config"])


def _validate_model_state(payload,model):
    import torch
    if payload["model_class"]!=_class_name(model):
        raise ValueError("checkpoint model class does not match")
    if payload["model_config"]!=_model_config(model):
        raise ValueError("checkpoint model configuration does not match; rebuild the model with the saved configuration")
    current=model.state_dict()
    saved=payload["model_state_dict"]
    if set(current)!=set(saved):
        raise ValueError("checkpoint model state keys do not match")
    for name,value in current.items():
        other=saved[name]
        if not isinstance(value,torch.Tensor) or not isinstance(other,torch.Tensor):
            raise ValueError("checkpoint model states must contain tensors")
        if value.shape!=other.shape or value.dtype!=other.dtype or value.layout!=other.layout:
            raise ValueError(f"checkpoint model tensor shape, dtype or layout does not match: {name}")


def _validate_states(payload,model,optimizer):
    import torch
    from tiny_lm.optim import AdamW,_check_group
    _validate_model_state(payload,model)
    if payload["optimizer_class"]!=_class_name(optimizer):
        raise ValueError("checkpoint optimizer class does not match")
    if payload["optimizer_parameter_names"]!=_parameter_names(model,optimizer):
        raise ValueError("checkpoint optimizer parameter names or ordering do not match")
    state=payload["optimizer_state_dict"]
    if set(state)!={"state","param_groups"} or not isinstance(state["state"],dict) or not isinstance(state["param_groups"],list):
        raise ValueError("checkpoint optimizer state is invalid")
    if len(state["param_groups"])!=len(optimizer.param_groups):
        raise ValueError("checkpoint optimizer parameter group count does not match")
    saved_ids=set()
    for saved_group,group in zip(state["param_groups"],optimizer.param_groups,strict=True):
        if not isinstance(saved_group,dict) or not isinstance(saved_group.get("params"),list) or len(saved_group["params"])!=len(group["params"]):
            raise ValueError("checkpoint optimizer parameter group sizes do not match")
        if isinstance(optimizer,AdamW):
            try:
                _check_group(saved_group)
            except (KeyError,TypeError,ValueError) as exc:
                raise ValueError("checkpoint AdamW hyperparameters are invalid") from exc
        for key,p in zip(saved_group["params"],group["params"],strict=True):
            if type(key) is not int or key in saved_ids:
                raise ValueError("checkpoint optimizer parameter IDs must be unique integers")
            saved_ids.add(key)
            entry=state["state"].get(key,{})
            if not isinstance(entry,dict):
                raise ValueError("checkpoint optimizer parameter state must be a dictionary")
            if isinstance(optimizer,AdamW) and entry:
                if set(entry)!={"step","exp_avg","exp_avg_sq"} or type(entry.get("step")) is not int or not 0<=entry["step"]<=payload["iteration"]:
                    raise ValueError("checkpoint AdamW step or state fields are invalid")
                expected_dtype=torch.float32 if p.dtype in (torch.float16,torch.bfloat16) else p.dtype
                for name in ("exp_avg","exp_avg_sq"):
                    value=entry[name]
                    if not isinstance(value,torch.Tensor) or value.shape!=p.shape or value.dtype!=expected_dtype or value.layout!=torch.strided:
                        raise ValueError(f"checkpoint AdamW {name} shape or dtype does not match")
    if set(state["state"])-saved_ids:
        raise ValueError("checkpoint optimizer state contains unknown parameter IDs")


def read_checkpoint(src)->dict:
    """
    在 CPU 上读取带版本的 checkpoint，供调用方查看配置并构造匹配模型。
    使用 weights_only=True；不自动执行旧格式迁移或恢复任何运行状态。
    """
    import torch
    try:
        payload=torch.load(src,map_location="cpu",weights_only=True)
    except (EOFError,IndexError,pickle.UnpicklingError) as exc:
        raise ValueError("checkpoint file is invalid, truncated or unsupported by weights_only loading") from exc
    _validate_payload(payload)
    return payload


def save_checkpoint(model,optimizer,iteration:int,out,*,config=None,generators=None):
    """
    保存已完成 iteration 次 optimizer 更新后的状态，不保存待累积梯度。
    generators 为具名 CPU 采样器；路径保存成功后原子替换，文件流由调用方管理。
    """
    import torch
    names=_parameter_names(model,optimizer)
    generators=_generators(generators)
    config=_config_dict(config)
    model_config=_model_config(model)
    _check_config(model_config,config)
    payload=dict(format_version=FORMAT_VERSION,iteration=iteration,model_class=_class_name(model),model_config=model_config,model_state_dict=model.state_dict(),optimizer_class=_class_name(optimizer),optimizer_parameter_names=names,optimizer_state_dict=optimizer.state_dict(),config=config,rng_state=_rng_state(generators))
    _validate_payload(payload)
    _validate_states(payload,model,optimizer)
    if isinstance(out,(str,os.PathLike)):
        target=Path(out).absolute()
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode="wb",prefix="."+target.name+".",suffix=".tmp",dir=target.parent,delete=False) as stream:
                temporary=Path(stream.name)
                torch.save(payload,stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary,target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    else:
        if isinstance(out,io.TextIOBase) or not callable(getattr(out,"write",None)):
            raise TypeError("out must be a filesystem path or writable binary stream")
        torch.save(payload,out)


def load_checkpoint(src,model,optimizer,*,config=None,generators=None,restore_rng:bool=True)->int:
    """
    恢复到现有模型和 optimizer，返回已完成更新次数；下一步使用该次数计算学习率。
    默认恢复全局与具名随机状态；只加载权重/optimizer 时可关闭随机状态恢复。
    成功后清除旧梯度，保留调用方的 train/eval 模式，不自动开始训练。
    """
    if type(restore_rng) is not bool:
        raise TypeError("restore_rng must be a bool")
    _parameter_names(model,optimizer)
    generators=_generators(generators)
    expected_config=_config_dict(config)
    payload=read_checkpoint(src)
    if expected_config is not None and payload["config"]!=expected_config:
        raise ValueError("checkpoint experiment configuration does not match")
    _validate_states(payload,model,optimizer)
    if restore_rng:
        _validate_rng(payload["rng_state"],generators)
    old_model=copy.deepcopy(model.state_dict())
    old_optimizer=copy.deepcopy(optimizer.state_dict())
    old_rng=_rng_state(generators) if restore_rng else None
    try:
        model.load_state_dict(payload["model_state_dict"],strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if restore_rng:
            _restore_rng(payload["rng_state"],generators)
    except Exception:
        model.load_state_dict(old_model,strict=True)
        optimizer.load_state_dict(old_optimizer)
        if old_rng is not None:
            _restore_rng(old_rng,generators)
        raise
    model.zero_grad(set_to_none=True)
    return payload["iteration"]
