"""
数组训练与验证的统计、模式恢复、loss 下降及精确续训测试。
"""

import importlib
import math
import random
from dataclasses import replace
import pytest

torch=pytest.importorskip("torch",reason="training tests require compute dependencies")
np=pytest.importorskip("numpy",reason="training tests require NumPy")
training=importlib.import_module("tiny_lm.train")
evaluation=importlib.import_module("tiny_lm.evaluate")
config_module=importlib.import_module("tiny_lm.config")
TransformerLM=importlib.import_module("tiny_lm.model.transformer").TransformerLM
checkpoint=importlib.import_module("tiny_lm.checkpoint")
get_batch=importlib.import_module("tiny_lm.data").get_batch
schedule=importlib.import_module("tiny_lm.schedule").get_lr_cosine_schedule


@pytest.fixture(autouse=True)
def preserve_rng():
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    yield
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def _config(tmp_path,**overrides):
    model=config_module.ModelConfig(vocab_size=8,context_length=4,d_model=8,num_layers=1,num_heads=2,d_ff=16,rope_theta=10000.0,norm_mode="pre_norm",ffn_type="swiglu")
    options=dict(batch_size=4,max_steps=30,learning_rate=0.03,min_learning_rate=0.003,warmup_steps=2,weight_decay=0.01,grad_clip=1.0,seed=42,device="cpu",eval_interval=5,save_interval=10)
    options.update(overrides)
    train=config_module.TrainingConfig(**options)
    data=config_module.DataConfig(train_data_path=tmp_path/"unused-train.bin",val_data_path=tmp_path/"unused-val.bin",tokenizer_path=tmp_path/"unused-tokenizer.json")
    return config_module.ProjectConfig(model=model,training=train,data=data,output=config_module.OutputConfig(output_dir=tmp_path/"run"))


def _data():
    return np.arange(256,dtype=np.int64)%8,(np.arange(192,dtype=np.int64)+3)%8


def _equal(left,right):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:
            _equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right,strict=True):
            _equal(a,b)
    else:
        assert left==right


def test_small_sequence_loss_decreases(tmp_path):
    config=_config(tmp_path)
    train_data,val_data=_data()
    initial=training.train(config,train_data,val_data,stop_after=0,save_checkpoints=False)
    before=evaluation.evaluate(initial.model,val_data,batch_size=4,context_length=4,num_batches=4,generator=torch.Generator().manual_seed(7)).loss
    result=training.train(config,train_data,val_data,eval_batches=2,save_checkpoints=False)
    after=evaluation.evaluate(result.model,val_data,batch_size=4,context_length=4,num_batches=4,generator=torch.Generator().manual_seed(7)).loss
    assert after<before*0.5
    assert result.completed_updates==30
    assert len(result.history)==30
    assert all(math.isfinite(row.train_loss) for row in result.history)
    assert result.history[-1].validation_loss is not None
    assert result.checkpoints==[]
    assert not config.output.output_dir.exists()
    assert not config.data.train_data_path.exists()


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32,torch.float64])
@pytest.mark.parametrize("split",[3,5])
def test_train_resume_exactly_matches_uninterrupted(tmp_path,dtype,split):
    config=_config(tmp_path,max_steps=8,eval_interval=3,save_interval=4,learning_rate=0.01)
    train_data,val_data=_data()
    full=training.train(config,train_data,val_data,dtype=dtype,eval_batches=2,save_checkpoints=False)
    part=training.train(config,train_data,val_data,dtype=dtype,eval_batches=2,stop_after=split)
    resumed=training.train(config,train_data,val_data,dtype=dtype,eval_batches=2,resume_from=part.checkpoints[-1])
    assert part.history+resumed.history==full.history
    assert resumed.completed_updates==8
    _equal(resumed.model.state_dict(),full.model.state_dict())
    _equal(resumed.optimizer.state_dict(),full.optimizer.state_dict())
    for name in full.generators:
        _equal(resumed.generators[name].get_state(),full.generators[name].get_state())
    saved=checkpoint.read_checkpoint(resumed.checkpoints[-1])
    assert saved["iteration"]==8
    assert saved["config"]==config.to_dict()
    assert saved["rng_state"]["generators"].keys()=={"train","validation"}


def test_stop_after_preserves_schedule_and_evaluation_cadence(tmp_path):
    config=_config(tmp_path,max_steps=10,eval_interval=4,save_interval=3)
    result=training.train(config,*_data(),eval_batches=1,stop_after=7)
    assert [path.name for path in result.checkpoints]==["step_00000003.pt","step_00000006.pt","step_00000007.pt"]
    assert [row.step for row in result.history if row.validation_loss is not None]==[4]
    assert [row.learning_rate for row in result.history]==[schedule(i,0.03,0.003,2,10) for i in range(7)]
    assert result.history[0].learning_rate==0
    assert result.history[2].learning_rate==0.03
    resumed=training.train(config,*_data(),eval_batches=1,resume_from=result.checkpoints[-1])
    assert [row.step for row in resumed.history if row.validation_loss is not None]==[8,10]
    assert all(state["step"]==10 for state in resumed.optimizer.state.values())


def test_validation_frequency_does_not_change_training(tmp_path):
    config=_config(tmp_path,max_steps=6,eval_interval=1)
    a=training.train(config,*_data(),eval_batches=1,save_checkpoints=False)
    b=training.train(replace(config,training=replace(config.training,eval_interval=4)),*_data(),eval_batches=3,save_checkpoints=False)
    assert [row.train_loss for row in a.history]==[row.train_loss for row in b.history]
    _equal(a.model.state_dict(),b.model.state_dict())
    _equal(a.optimizer.state_dict(),b.optimizer.state_dict())
    _equal(a.generators["train"].get_state(),b.generators["train"].get_state())


def test_evaluation_matches_token_weighted_reference_and_preserves_modes_gradients(tmp_path,monkeypatch):
    model=TransformerLM.from_config(_config(tmp_path).model,dtype=torch.float64)
    model.train()
    model.layers[0].eval()
    modes=[module.training for module in model.modules()]
    for p in model.parameters():
        p.grad=torch.ones_like(p)
    gradients=[p.grad.clone() for p in model.parameters()]
    global_state=torch.get_rng_state().clone()
    generator=torch.Generator().manual_seed(10)
    clone=torch.Generator().manual_seed(10)
    original=model.forward
    seen=[]
    def forward(x):
        seen.append((torch.is_grad_enabled(),[module.training for module in model.modules()]))
        return original(x)
    monkeypatch.setattr(model,"forward",forward)
    data=_data()[1]
    result=evaluation.evaluate(model,data,batch_size=2,context_length=3,num_batches=3,generator=generator)
    assert all(not grad and not any(flags) for grad,flags in seen)
    assert [module.training for module in model.modules()]==modes
    _equal([p.grad for p in model.parameters()],gradients)
    _equal(torch.get_rng_state(),global_state)
    losses=[]
    with torch.no_grad():
        for _ in range(3):
            x,y=get_batch(data,2,3,"cpu",generator=clone)
            losses.append(torch.nn.functional.cross_entropy(original(x).reshape(-1,8),y.reshape(-1),reduction="sum").item())
    assert result.loss==pytest.approx(math.fsum(losses)/18,rel=1e-12)
    assert result.num_tokens==18
    assert result.num_batches==3
    _equal(generator.get_state(),clone.get_state())


def test_evaluation_restores_each_mode_after_failure(tmp_path,monkeypatch):
    model=TransformerLM.from_config(_config(tmp_path).model)
    model.train()
    model.layers[0].eval()
    modes=[module.training for module in model.modules()]
    def fail(x):
        assert not torch.is_grad_enabled()
        assert not model.training
        raise RuntimeError("simulated forward failure")
    monkeypatch.setattr(model,"forward",fail)
    with pytest.raises(RuntimeError,match="simulated"):
        evaluation.evaluate(model,_data()[1],batch_size=2,context_length=4,num_batches=1,generator=torch.Generator())
    assert [module.training for module in model.modules()]==modes


@pytest.mark.parametrize("data",[np.arange(4),np.array([0,1,2,3,8]),np.array([0,1,2,3,-1]),np.zeros((2,4),dtype=np.int64),np.zeros(6),np.array([0,1,2,3,2**64-1],dtype=np.uint64),[0,1,2,3,4]])
def test_invalid_training_data_fails_before_output_or_rng_changes(tmp_path,data):
    config=_config(tmp_path)
    state=torch.get_rng_state().clone()
    with pytest.raises((TypeError,ValueError)):
        training.train(config,data,_data()[1])
    assert not config.output.output_dir.exists()
    _equal(torch.get_rng_state(),state)


@pytest.mark.parametrize("kwargs",[{"stop_after":-1},{"stop_after":31},{"stop_after":True},{"eval_batches":0},{"eval_batches":False},{"save_checkpoints":1},{"dtype":torch.int64}])
def test_invalid_training_options(tmp_path,kwargs):
    config=_config(tmp_path)
    with pytest.raises((TypeError,ValueError)):
        training.train(config,*_data(),**kwargs)
    assert not config.output.output_dir.exists()


def test_resume_rejects_changed_config_and_past_requested_limit(tmp_path):
    config=_config(tmp_path,max_steps=6)
    part=training.train(config,*_data(),stop_after=3,eval_batches=1)
    with pytest.raises(ValueError,match="configuration"):
        training.train(replace(config,training=replace(config.training,learning_rate=0.02)),*_data(),resume_from=part.checkpoints[-1],eval_batches=1)
    with pytest.raises(ValueError,match="exceeds"):
        training.train(config,*_data(),resume_from=part.checkpoints[-1],stop_after=2,eval_batches=1)


def test_completed_checkpoint_is_a_noop(tmp_path):
    config=_config(tmp_path,max_steps=3)
    result=training.train(config,*_data(),eval_batches=1)
    resumed=training.train(config,*_data(),eval_batches=1,resume_from=result.checkpoints[-1])
    assert resumed.completed_updates==3
    assert resumed.history==[]
    assert resumed.checkpoints==[]
    _equal(resumed.model.state_dict(),result.model.state_dict())
    _equal(resumed.optimizer.state_dict(),result.optimizer.state_dict())


@pytest.mark.parametrize("failure",["loss","gradient","parameter"])
def test_nonfinite_training_stops_without_saving_bad_state(tmp_path,monkeypatch,failure):
    config=_config(tmp_path,max_steps=3,save_interval=1)
    losses=importlib.import_module("tiny_lm.losses")
    optim=importlib.import_module("tiny_lm.optim")
    original=losses.cross_entropy
    if failure=="parameter":
        step=optim.AdamW.step
        def bad_step(self,closure=None):
            result=step(self,closure)
            with torch.no_grad():
                self.param_groups[0]["params"][0].fill_(float("inf"))
            return result
        monkeypatch.setattr(optim.AdamW,"step",bad_step)
    else:
        def bad_loss(logits,targets):
            result=original(logits,targets)
            if failure=="loss":
                return result*float("nan")
            result.register_hook(lambda gradient:gradient*float("nan"))
            return result
        monkeypatch.setattr(losses,"cross_entropy",bad_loss)
    with pytest.raises((FloatingPointError,ValueError),match="finite"):
        training.train(config,*_data(),eval_batches=1)
    assert not list(config.output.output_dir.glob("*.pt"))


@pytest.mark.parametrize("kwargs",[{"batch_size":0},{"context_length":0},{"context_length":5},{"num_batches":0},{"num_batches":True},{"generator":None},{"generator":torch.default_generator}])
def test_invalid_evaluation_options(tmp_path,kwargs):
    model=TransformerLM.from_config(_config(tmp_path).model)
    options=dict(batch_size=2,context_length=4,num_batches=2,generator=torch.Generator())
    options.update(kwargs)
    with pytest.raises((TypeError,ValueError)):
        evaluation.evaluate(model,_data()[1],**options)


def test_evaluation_rejects_nonfinite_loss(tmp_path,monkeypatch):
    model=TransformerLM.from_config(_config(tmp_path).model)
    model.train()
    def invalid(x):
        return torch.full((*x.shape,8),float("nan"))
    monkeypatch.setattr(model,"forward",invalid)
    with pytest.raises(FloatingPointError,match="finite"):
        evaluation.evaluate(model,_data()[1],batch_size=2,context_length=4,num_batches=1,generator=torch.Generator())
    assert model.training


def test_clipping_is_applied_before_update(tmp_path,monkeypatch):
    config=_config(tmp_path,max_steps=3,grad_clip=0.01)
    optim=importlib.import_module("tiny_lm.optim")
    original=optim.AdamW.step
    norms=[]
    def checked(self,closure=None):
        norm=torch.linalg.vector_norm(torch.cat([p.grad.flatten() for group in self.param_groups for p in group["params"]]))
        norms.append(norm.item())
        return original(self,closure)
    monkeypatch.setattr(optim.AdamW,"step",checked)
    training.train(config,*_data(),eval_batches=1,save_checkpoints=False)
    assert len(norms)==3
    assert all(0<value<=0.010001 for value in norms)
