"""
Checkpoint 的精确续训、配置检查、随机状态与失败保护。
"""

import copy
import importlib
import io
import random
from dataclasses import replace
import pytest

torch=pytest.importorskip("torch",reason="checkpoint tests require compute dependencies")
np=pytest.importorskip("numpy",reason="checkpoint RNG tests require NumPy")
checkpoint=importlib.import_module("tiny_lm.checkpoint")
TransformerLM=importlib.import_module("tiny_lm.model.transformer").TransformerLM
AdamW=importlib.import_module("tiny_lm.optim").AdamW
get_batch=importlib.import_module("tiny_lm.data").get_batch
cross_entropy=importlib.import_module("tiny_lm.losses").cross_entropy
get_lr=importlib.import_module("tiny_lm.schedule").get_lr_cosine_schedule
ModelConfig=importlib.import_module("tiny_lm.config").ModelConfig


@pytest.fixture(autouse=True)
def preserve_global_rng():
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    yield
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def _model(dtype=torch.float32):
    return TransformerLM(11,6,4,1,2,7,10000.0,dtype=dtype)


def _config():
    return ModelConfig(vocab_size=11,context_length=6,d_model=4,num_layers=1,num_heads=2,d_ff=7,rope_theta=10000.0,norm_mode="pre_norm",ffn_type="swiglu")


def _assert_equal(left,right):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,np.ndarray):
        np.testing.assert_array_equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:
            _assert_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right,strict=True):
            _assert_equal(a,b)
    else:
        assert left==right


def _bytes(payload):
    stream=io.BytesIO()
    torch.save(payload,stream)
    stream.seek(0)
    return stream


def _save(model,optimizer,iteration=0,**kwargs):
    stream=io.BytesIO()
    checkpoint.save_checkpoint(model,optimizer,iteration,stream,**kwargs)
    stream.seek(0)
    return stream


def _step(model,optimizer,generator,step):
    x,y=get_batch(np.arange(44,dtype=np.int64)%11,2,3,"cpu",generator=generator)
    lr=get_lr(step,0.002,0.0002,1,8)
    optimizer.param_groups[0]["lr"]=lr
    optimizer.zero_grad()
    factor=0.9+0.1*random.random()+0.01*np.random.random()+0.01*torch.rand(()).item()
    loss=cross_entropy(model(x),y)*factor
    loss.backward()
    optimizer.step()
    return loss.detach().clone(),x.clone(),lr


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32,torch.float64])
@pytest.mark.parametrize("use_path",[False,True])
def test_resume_matches_uninterrupted_training(tmp_path,dtype,use_path):
    torch.manual_seed(12)
    random.seed(13)
    np.random.seed(14)
    model=_model(dtype)
    optimizer=AdamW(model.parameters())
    train=torch.Generator().manual_seed(15)
    validation=torch.Generator().manual_seed(16)
    generators={"train":train,"validation":validation}
    for step in range(2):
        _step(model,optimizer,train,step)
    target=tmp_path/"state.pt" if use_path else io.BytesIO()
    checkpoint.save_checkpoint(model,optimizer,2,target,config=_config(),generators=generators)
    expected=[_step(model,optimizer,train,step) for step in range(2,5)]
    expected_validation=torch.rand(3,generator=validation)
    expected_rng=(random.random(),np.random.random(),torch.rand(3))
    expected_model=copy.deepcopy(model.state_dict())
    expected_optimizer=copy.deepcopy(optimizer.state_dict())
    if not use_path:
        target.seek(0)
    metadata=checkpoint.read_checkpoint(target)
    resumed=TransformerLM(**metadata["model_config"],dtype=dtype)
    resumed_optimizer=AdamW(resumed.parameters(),lr=0.99)
    resumed_train=torch.Generator().manual_seed(100)
    resumed_validation=torch.Generator().manual_seed(101)
    if not use_path:
        target.seek(0)
    completed=checkpoint.load_checkpoint(target,resumed,resumed_optimizer,config=_config(),generators={"train":resumed_train,"validation":resumed_validation})
    assert completed==2
    actual=[_step(resumed,resumed_optimizer,resumed_train,step) for step in range(completed,5)]
    _assert_equal(actual,expected)
    _assert_equal(resumed.state_dict(),expected_model)
    _assert_equal(resumed_optimizer.state_dict(),expected_optimizer)
    _assert_equal(torch.rand(3,generator=resumed_validation),expected_validation)
    _assert_equal((random.random(),np.random.random(),torch.rand(3)),expected_rng)
    expected_dtype=torch.float64 if dtype==torch.float64 else torch.float32
    assert all(state["exp_avg"].dtype==expected_dtype for state in resumed_optimizer.state.values())
    if use_path:
        assert list(tmp_path.iterdir())==[target]


def test_zero_step_clears_old_optimizer_state_and_gradients_but_preserves_mode():
    model=_model()
    optimizer=AdamW(model.parameters())
    stream=_save(model,optimizer)
    _step(model,optimizer,None,0)
    model.eval()
    assert optimizer.state
    assert any(p.grad is not None for p in model.parameters())
    assert checkpoint.load_checkpoint(stream,model,optimizer)==0
    assert not optimizer.state
    assert not model.training
    assert all(p.grad is None for p in model.parameters())


def test_restore_rng_false_keeps_current_rng_and_allows_omitted_generators():
    model=_model()
    optimizer=AdamW(model.parameters())
    stream=_save(model,optimizer,generators={"train":torch.Generator()})
    state=checkpoint._rng_state({})
    checkpoint.load_checkpoint(stream,model,optimizer,restore_rng=False)
    _assert_equal(checkpoint._rng_state({}),state)


@pytest.mark.parametrize("change",["version","legacy","iteration","model_class","optimizer_class","model_config","tensor_shape","tensor_dtype","model_key","parameter_order","group_size","group_lr","moment_shape","moment_dtype","step","unknown_parameter","rng","generator_names","config"])
def test_invalid_checkpoint_rejected_without_changing_runtime(change):
    model=_model()
    optimizer=AdamW(model.parameters())
    _step(model,optimizer,None,0)
    payload=checkpoint.read_checkpoint(_save(model,optimizer,1,config=_config()))
    if change=="version":
        payload["format_version"]=99
    elif change=="legacy":
        del payload["format_version"]
    elif change=="iteration":
        payload["iteration"]=True
    elif change=="model_class":
        payload["model_class"]="different.Model"
    elif change=="optimizer_class":
        payload["optimizer_class"]="different.Optimizer"
    elif change=="model_config":
        payload["model_config"]["rope_theta"]=5000.0
    elif change=="tensor_shape":
        payload["model_state_dict"]["lm_head.weight"]=torch.zeros(1)
    elif change=="tensor_dtype":
        payload["model_state_dict"]["lm_head.weight"]=payload["model_state_dict"]["lm_head.weight"].double()
    elif change=="model_key":
        del payload["model_state_dict"]["lm_head.weight"]
    elif change=="parameter_order":
        payload["optimizer_parameter_names"][0].reverse()
    elif change=="group_size":
        payload["optimizer_state_dict"]["param_groups"][0]["params"].pop()
    elif change=="group_lr":
        payload["optimizer_state_dict"]["param_groups"][0]["lr"]=float("nan")
    elif change=="moment_shape":
        payload["optimizer_state_dict"]["state"][0]["exp_avg"]=torch.zeros(1)
    elif change=="moment_dtype":
        payload["optimizer_state_dict"]["state"][0]["exp_avg"]=payload["optimizer_state_dict"]["state"][0]["exp_avg"].half()
    elif change=="step":
        payload["optimizer_state_dict"]["state"][0]["step"]=2
    elif change=="unknown_parameter":
        payload["optimizer_state_dict"]["state"][10000]={}
    elif change=="rng":
        payload["rng_state"]["torch_cpu"]=torch.zeros(3)
    elif change=="generator_names":
        payload["rng_state"]["generators"]={"missing":torch.get_rng_state()}
    else:
        payload["config"]["model"]["d_ff"]=123
    old_model=copy.deepcopy(model.state_dict())
    old_optimizer=copy.deepcopy(optimizer.state_dict())
    old_rng=checkpoint._rng_state({})
    with pytest.raises(ValueError):
        checkpoint.load_checkpoint(_bytes(payload),model,optimizer)
    _assert_equal(model.state_dict(),old_model)
    _assert_equal(optimizer.state_dict(),old_optimizer)
    _assert_equal(checkpoint._rng_state({}),old_rng)


def test_reject_same_shape_rope_change_and_reordered_live_optimizer():
    model=_model()
    optimizer=AdamW(model.parameters())
    payload=checkpoint.read_checkpoint(_save(model,optimizer))
    other=TransformerLM(11,6,4,1,2,7,5000.0)
    with pytest.raises(ValueError,match="model configuration"):
        checkpoint.load_checkpoint(_bytes(payload),other,AdamW(other.parameters()))
    reversed_optimizer=AdamW(list(model.parameters())[::-1])
    with pytest.raises(ValueError,match="ordering"):
        checkpoint.load_checkpoint(_bytes(payload),model,reversed_optimizer)


def test_experiment_config_checked_and_model_config_copy_is_independent():
    model=_model()
    optimizer=AdamW(model.parameters())
    config=_config()
    with pytest.raises(ValueError,match="config.model"):
        _save(model,optimizer,config=replace(config,rope_theta=5000.0))
    stream=_save(model,optimizer,config=config)
    with pytest.raises(ValueError,match="experiment configuration"):
        checkpoint.load_checkpoint(stream,model,optimizer,config=replace(config,rope_theta=5000.0))
    info=model.get_config()
    info["rope_theta"]=5000.0
    assert model.get_config()["rope_theta"]==10000.0


@pytest.mark.parametrize("failure",["serialize","replace"])
def test_atomic_save_keeps_previous_file_and_removes_temporary(tmp_path,monkeypatch,failure):
    model=_model()
    optimizer=AdamW(model.parameters())
    target=tmp_path/"checkpoint.pt"
    checkpoint.save_checkpoint(model,optimizer,0,target)
    original=target.read_bytes()
    def fail(*args,**kwargs):
        if failure=="serialize":
            args[1].write(b"partial")
        raise OSError("simulated write failure")
    if failure=="serialize":
        monkeypatch.setattr(torch,"save",fail)
    else:
        monkeypatch.setattr(checkpoint.os,"replace",fail)
    with pytest.raises(OSError,match="simulated"):
        checkpoint.save_checkpoint(model,optimizer,0,target)
    assert target.read_bytes()==original
    assert list(tmp_path.iterdir())==[target]


@pytest.mark.parametrize("stage",["optimizer","rng"])
def test_restore_failure_rolls_back_live_state(monkeypatch,stage):
    model=_model()
    optimizer=AdamW(model.parameters())
    stream=_save(model,optimizer)
    _step(model,optimizer,None,0)
    old_model=copy.deepcopy(model.state_dict())
    old_optimizer=copy.deepcopy(optimizer.state_dict())
    old_gradients=[p.grad.clone() for p in model.parameters()]
    old_rng=checkpoint._rng_state({})
    calls=0
    original=optimizer.load_state_dict if stage=="optimizer" else checkpoint._restore_rng
    def fail_once(*args):
        nonlocal calls
        calls+=1
        original(*args)
        if calls==1:
            raise RuntimeError("simulated restore failure")
    if stage=="optimizer":
        monkeypatch.setattr(optimizer,"load_state_dict",fail_once)
    else:
        monkeypatch.setattr(checkpoint,"_restore_rng",fail_once)
    with pytest.raises(RuntimeError,match="simulated"):
        checkpoint.load_checkpoint(stream,model,optimizer)
    _assert_equal(model.state_dict(),old_model)
    _assert_equal(optimizer.state_dict(),old_optimizer)
    _assert_equal([p.grad for p in model.parameters()],old_gradients)
    _assert_equal(checkpoint._rng_state({}),old_rng)


@pytest.mark.parametrize("iteration",[-1,True,1.5])
def test_invalid_iteration(iteration):
    model=_model()
    with pytest.raises(ValueError,match="iteration"):
        _save(model,AdamW(model.parameters()),iteration)


def test_save_rejects_incorrect_completed_count():
    model=_model()
    optimizer=AdamW(model.parameters())
    _step(model,optimizer,None,0)
    with pytest.raises(ValueError,match="step"):
        _save(model,optimizer,0)


def test_invalid_generators_and_foreign_optimizer():
    model=_model()
    optimizer=AdamW(model.parameters())
    generator=torch.Generator()
    for value in ({"":generator},{"train":1},{"a":generator,"b":generator},[]):
        with pytest.raises((TypeError,ValueError),match="generator"):
            _save(model,optimizer,generators=value)
    with pytest.raises(ValueError,match="belong"):
        _save(model,AdamW(_model().parameters()))


def test_generic_module_sgd_and_restricted_load(monkeypatch):
    model=torch.nn.Linear(2,2)
    optimizer=torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9)
    model(torch.ones(1,2)).sum().backward()
    optimizer.step()
    stream=_save(model,optimizer,1)
    original=torch.load
    def checked(*args,**kwargs):
        assert kwargs["weights_only"] is True
        assert kwargs["map_location"]=="cpu"
        return original(*args,**kwargs)
    monkeypatch.setattr(torch,"load",checked)
    restored=torch.nn.Linear(2,2)
    restored_optimizer=torch.optim.SGD(restored.parameters(),lr=0.01)
    assert checkpoint.load_checkpoint(stream,restored,restored_optimizer)==1
    _assert_equal(restored.state_dict(),model.state_dict())
    _assert_equal(restored_optimizer.state_dict(),optimizer.state_dict())


def test_missing_parent_and_text_stream_errors(tmp_path):
    model=_model()
    optimizer=AdamW(model.parameters())
    with pytest.raises(FileNotFoundError):
        checkpoint.save_checkpoint(model,optimizer,0,tmp_path/"absent"/"state.pt")
    with pytest.raises(TypeError,match="binary"):
        checkpoint.save_checkpoint(model,optimizer,0,io.StringIO())


def test_full_project_config_and_replacing_existing_checkpoint(tmp_path):
    from pathlib import Path
    from tiny_lm.config import load_config
    project=load_config(Path(__file__).resolve().parents[1]/"configs"/"smoke.yaml")
    project=replace(project,model=_config())
    model=TransformerLM.from_config(project.model)
    optimizer=AdamW(model.parameters())
    target=tmp_path/"latest.pt"
    checkpoint.save_checkpoint(model,optimizer,0,target,config=project)
    _step(model,optimizer,None,0)
    checkpoint.save_checkpoint(model,optimizer,1,target,config=project)
    metadata=checkpoint.read_checkpoint(target)
    assert metadata["config"]==project.to_dict()
    assert metadata["iteration"]==1
    assert checkpoint.load_checkpoint(target,model,optimizer,config=project)==1
    assert list(tmp_path.iterdir())==[target]


@pytest.mark.parametrize("kind",["python","numpy","generators"])
def test_invalid_random_states_do_not_change_global_rng(kind):
    model=_model()
    optimizer=AdamW(model.parameters())
    generator=torch.Generator().manual_seed(7)
    payload=checkpoint.read_checkpoint(_save(model,optimizer,generators={"train":generator}))
    if kind=="generators":
        payload["rng_state"][kind]["train"]=torch.zeros(1)
    else:
        payload["rng_state"][kind]=("invalid",)
    before=checkpoint._rng_state({"train":generator})
    with pytest.raises(ValueError,match="RNG"):
        checkpoint.load_checkpoint(_bytes(payload),model,optimizer,generators={"train":generator})
    _assert_equal(checkpoint._rng_state({"train":generator}),before)


def test_cuda_rng_requires_matching_runtime_when_restored():
    if torch.cuda.is_available():
        pytest.skip("CPU portability check requires a CPU-only runtime")
    model=_model()
    optimizer=AdamW(model.parameters())
    payload=checkpoint.read_checkpoint(_save(model,optimizer))
    payload["rng_state"]["torch_cuda"]=[torch.zeros(8,dtype=torch.uint8)]
    with pytest.raises(ValueError,match="CUDA RNG device count"):
        checkpoint.load_checkpoint(_bytes(payload),model,optimizer)
    assert checkpoint.load_checkpoint(_bytes(payload),model,optimizer,restore_rng=False)==0
