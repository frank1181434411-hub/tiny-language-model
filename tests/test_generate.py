"""
自回归生成的采样对照、状态保护、边界与模型专用加载测试。
"""

import importlib
import io
from dataclasses import replace
import pytest
from tiny_lm.config import ModelConfig
from tiny_lm.checkpoint import save_checkpoint,read_checkpoint
from tiny_lm.generate import generate,load_generation_model,_sampling_probabilities

torch=pytest.importorskip("torch")
TransformerLM=importlib.import_module("tiny_lm.model.transformer").TransformerLM
AdamW=importlib.import_module("tiny_lm.optim").AdamW


@pytest.fixture(autouse=True)
def preserve_rng():
    state=torch.get_rng_state()
    yield
    torch.set_rng_state(state)


def model_config():
    return ModelConfig(vocab_size=8,context_length=3,d_model=8,num_layers=1,num_heads=2,d_ff=16,norm_mode="pre_norm",ffn_type="swiglu",rope_theta=10000.0)


def model(dtype=torch.float32):
    return TransformerLM.from_config(model_config(),dtype=dtype)


def generator(seed=17):
    return torch.Generator().manual_seed(seed)


class ScriptedLM(TransformerLM):
    def __init__(self,rows):
        super().__init__(vocab_size=8,context_length=3,d_model=8,num_layers=1,num_heads=2,d_ff=16,rope_theta=10000.0)
        self.rows=rows
        self.inputs=[]

    def forward(self,ids):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in self.modules())
        self.inputs.append(ids.clone())
        row=self.rows[min(len(self.inputs)-1,len(self.rows)-1)]
        return row.expand(1,ids.shape[1],8)


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32,torch.float64])
@pytest.mark.parametrize("temperature,top_p,top_k",[(1.0,1.0,None),(0.8,1.0,None),(1.0,0.7,None),(1.0,1.0,3),(1.4,0.8,4),(0.1,0.01,1)])
def test_sampling_probabilities_match_independent_reference(dtype,temperature,top_p,top_k):
    logits=torch.tensor([[0.2,1.3,-0.7,2.4,0.1,-3.0,1.1,0.6]],dtype=dtype)
    expected=logits.to(torch.float64 if dtype==torch.float64 else torch.float32).clone()
    if temperature!=1:
        expected=expected/(temperature+1e-8)
    if top_k is not None:
        threshold=sorted(expected[0].tolist(),reverse=True)[top_k-1]
        expected=expected.masked_fill(expected<threshold,float("-inf"))
    if top_p<1:
        probabilities=torch.softmax(expected,dim=-1)[0]
        ordered=sorted(range(8),key=lambda i:expected[0,i].item(),reverse=True)
        cumulative=0.0
        keep=[]
        for index in ordered:
            keep.append(index)
            cumulative+=probabilities[index].item()
            if cumulative>top_p:
                break
        for index in set(range(8))-set(keep):
            expected[0,index]=float("-inf")
    expected=torch.softmax(expected,dim=-1)
    actual=_sampling_probabilities(logits,temperature,top_p,top_k)
    assert actual.dtype==expected.dtype
    torch.testing.assert_close(actual,expected)


def test_top_k_ties_remain_and_top_p_keeps_crossing_token():
    logits=torch.tensor([[3.,3.,0.,-1.,-2.,-3.,-4.,-5.]])
    probs=_sampling_probabilities(logits,1.,1.,1)
    assert probs[0,0]==probs[0,1]==0.5
    logits=torch.log(torch.tensor([[0.4,0.3,0.2,0.1,0.,0.,0.,0.]]).clamp_min(1e-20))
    probs=_sampling_probabilities(logits,1.,0.6,None)
    assert (probs>0).sum()==2


def test_context_cropping_eos_and_prompt_are_preserved():
    row=torch.arange(8,dtype=torch.float32)
    second=row.clone()
    second[2]=100.
    network=ScriptedLM([row,second])
    prompt=torch.tensor([[0,1,2,3,4]],dtype=torch.int32)
    output=generate(network,prompt,10,top_k=1,eos_token_id=2,generator=generator())
    assert output.tolist()==[[0,1,2,3,4,7,2]]
    assert prompt.tolist()==[[0,1,2,3,4]]
    assert [ids.tolist() for ids in network.inputs]==[[[2,3,4]],[[3,4,7]]]
    assert output.dtype==torch.int64


def test_existing_prompt_eos_does_not_stop_new_generation():
    network=ScriptedLM([torch.arange(8,dtype=torch.float32)])
    output=generate(network,torch.tensor([[2]]),2,eos_token_id=2,top_k=1,generator=generator())
    assert output.tolist()==[[2,7,7]]


def test_zero_tokens_does_not_forward_or_consume_rng():
    network=ScriptedLM([torch.zeros(8)])
    rng=generator()
    before=rng.get_state().clone()
    prompt=torch.tensor([[1,2]])
    output=generate(network,prompt,0,generator=rng)
    assert output.tolist()==prompt.tolist()
    assert output.data_ptr()!=prompt.data_ptr()
    assert network.inputs==[]
    assert torch.equal(rng.get_state(),before)


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_generation_matches_reference_and_preserves_state(dtype):
    network=model(dtype)
    network.train()
    network.layers[0].eval()
    modes=[module.training for module in network.modules()]
    weights={name:value.clone() for name,value in network.state_dict().items()}
    for parameter in network.parameters():
        parameter.grad=torch.ones_like(parameter)
    global_rng=torch.get_rng_state().clone()
    prompt=torch.tensor([[1,2,3,4]])
    output=generate(network,prompt,5,temperature=0.8,top_p=0.9,top_k=5,generator=generator())
    assert torch.equal(torch.get_rng_state(),global_rng)
    assert [module.training for module in network.modules()]==modes
    for name,value in network.state_dict().items():
        assert torch.equal(value,weights[name])
    assert all(torch.equal(p.grad,torch.ones_like(p)) for p in network.parameters())
    assert torch.equal(output,generate(network,prompt,5,temperature=0.8,top_p=0.9,top_k=5,generator=generator()))
    # 独立前向循环验证追加顺序与采样器状态，不通过 generate 调用。
    reference=prompt.clone()
    rng=generator()
    network.eval()
    with torch.no_grad():
        for _ in range(5):
            logits=network(reference[:,-3:])[:,-1,:]
            probs=_sampling_probabilities(logits,0.8,0.9,5)
            reference=torch.cat((reference,torch.multinomial(probs,1,generator=rng)),dim=1)
    assert torch.equal(output,reference)


@pytest.mark.parametrize("bad",[float("nan"),float("inf"),float("-inf")])
def test_nonfinite_logits_restore_modes(bad):
    row=torch.zeros(8)
    row[0]=bad
    network=ScriptedLM([row])
    network.layers[0].eval()
    modes=[module.training for module in network.modules()]
    with pytest.raises(FloatingPointError,match="finite"):
        generate(network,torch.tensor([[1]]),2,generator=generator())
    assert [module.training for module in network.modules()]==modes


@pytest.mark.parametrize("options",[{"max_new_tokens":-1},{"max_new_tokens":True},{"temperature":0},{"temperature":-1},{"temperature":float("nan")},{"temperature":float("inf")},{"temperature":True},{"temperature":10**1000},{"top_p":0},{"top_p":1.1},{"top_p":float("nan")},{"top_p":False},{"top_k":0},{"top_k":9},{"top_k":True},{"eos_token_id":-1},{"eos_token_id":8},{"eos_token_id":True}])
def test_invalid_sampling_parameters(options):
    arguments={"max_new_tokens":1,**options}
    with pytest.raises((ValueError,TypeError)):
        generate(model(),torch.tensor([[1]]),generator=generator(),**arguments)


@pytest.mark.parametrize("prompt",[[],torch.tensor([1]),torch.zeros((1,0),dtype=torch.int64),torch.tensor([[1],[2]]),torch.tensor([[1.]]),torch.tensor([[True]]),torch.tensor([[-1]]),torch.tensor([[8]])])
def test_invalid_prompts(prompt):
    with pytest.raises((ValueError,TypeError)):
        generate(model(),prompt,1,generator=generator())


@pytest.mark.parametrize("rng",[None,torch.default_generator,"seed"])
def test_invalid_generator(rng):
    with pytest.raises((ValueError,TypeError)):
        generate(model(),torch.tensor([[1]]),1,generator=rng)


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_load_generation_model_preserves_weights_dtype_and_rng(tmp_path,dtype):
    network=model(dtype)
    path=tmp_path/"model.pt"
    save_checkpoint(network,AdamW(network.parameters()),0,path,config=model_config())
    before=path.read_bytes()
    rng=torch.get_rng_state().clone()
    loaded=load_generation_model(path,config=model_config())
    assert not loaded.training
    assert loaded.token_embeddings.weight.dtype==dtype
    assert torch.equal(torch.get_rng_state(),rng)
    assert path.read_bytes()==before
    for name,value in network.state_dict().items():
        assert torch.equal(value,loaded.state_dict()[name])
    ids=torch.tensor([[1,2]])
    assert torch.equal(generate(network,ids,2,generator=generator()),generate(loaded,ids,2,generator=generator()))


@pytest.mark.parametrize("problem",["model_class","missing_config","extra_config","weight_dtype","weight_shape","weight_missing","nonfinite","embedding_dtype","requested_config"])
def test_model_loader_rejects_incompatible_checkpoint(problem):
    network=model()
    stream=io.BytesIO()
    save_checkpoint(network,AdamW(network.parameters()),0,stream)
    stream.seek(0)
    payload=read_checkpoint(stream)
    if problem=="model_class":
        payload["model_class"]="another.Model"
    elif problem=="missing_config":
        payload["model_config"]=None
    elif problem=="extra_config":
        payload["model_config"]["unexpected"]=True
    elif problem=="weight_dtype":
        payload["model_state_dict"]["lm_head.weight"]=payload["model_state_dict"]["lm_head.weight"].double()
    elif problem=="weight_shape":
        payload["model_state_dict"]["lm_head.weight"]=torch.zeros(1)
    elif problem=="weight_missing":
        del payload["model_state_dict"]["lm_head.weight"]
    elif problem=="nonfinite":
        payload["model_state_dict"]["lm_head.weight"][0,0]=float("nan")
    elif problem=="embedding_dtype":
        payload["model_state_dict"]["token_embeddings.weight"]=payload["model_state_dict"]["token_embeddings.weight"].long()
    target=io.BytesIO()
    torch.save(payload,target)
    target.seek(0)
    config=replace(model_config(),norm_mode="post_norm") if problem=="requested_config" else None
    with pytest.raises((ValueError,TypeError)):
        load_generation_model(target,config=config)
