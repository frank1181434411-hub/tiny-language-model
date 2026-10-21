"""
Check sampling boundaries, token conversion and independent random state.
"""

import pytest

from tiny_lm.data import get_batch

np=pytest.importorskip("numpy",reason="data tests require the optional compute dependencies")
torch=pytest.importorskip("torch",reason="data tests require the optional compute dependencies")


def test_windows_include_first_and_last_start_and_allow_repeats():
    dataset=np.arange(8,dtype=np.int64)
    x,y=get_batch(dataset,512,3,"cpu",generator=torch.Generator().manual_seed(42))
    assert x.shape==y.shape==(512,3)
    assert x.dtype==y.dtype==torch.int64
    assert x.device==y.device==torch.device("cpu")
    assert set(x[:,0].tolist())==set(range(5))
    torch.testing.assert_close(x,x[:,0,None]+torch.arange(3))
    torch.testing.assert_close(y,x+1)


def test_minimum_dataset_length_has_one_window():
    x,y=get_batch(np.array([7,2,9]),4,2,"cpu")
    torch.testing.assert_close(x,torch.tensor([[7,2]]*4))
    torch.testing.assert_close(y,torch.tensor([[2,9]]*4))


@pytest.mark.parametrize("dtype",[np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64])
def test_integer_dtypes_convert_without_changing_values(dtype):
    dataset=np.array([0,2,7,1],dtype=dtype)
    x,y=get_batch(dataset,1,3,torch.device("cpu"))
    torch.testing.assert_close(x,torch.tensor([[0,2,7]]))
    torch.testing.assert_close(y,torch.tensor([[2,7,1]]))


@pytest.mark.parametrize("dtype",[np.int64,np.uint64])
def test_int64_max_is_preserved_exactly(dtype):
    maximum=np.iinfo(np.int64).max
    x,y=get_batch(np.array([maximum,maximum,0],dtype=dtype),1,2,"cpu")
    assert x.tolist()==[[maximum,maximum]]
    assert y.tolist()==[[maximum,0]]


@pytest.mark.parametrize("dataset",[np.arange(12)[::2],np.arange(6)[::-1],np.arange(6,dtype=">i4")])
def test_strided_and_non_native_byte_order_arrays(dataset):
    x,y=get_batch(dataset,1,len(dataset)-1,"cpu")
    assert x.tolist()==[dataset[:-1].tolist()]
    assert y.tolist()==[dataset[1:].tolist()]
    assert x.is_contiguous() and y.is_contiguous()


def test_readonly_memmap_is_not_modified(tmp_path):
    path=tmp_path/"tokens.bin"
    original=np.array([1,4,2,7],dtype=np.uint16)
    path.write_bytes(original.tobytes())
    dataset=np.memmap(path,dtype=np.uint16,mode="r")
    try:
        x,y=get_batch(dataset,1,3,"cpu")
        assert x.tolist()==[[1,4,2]]
        assert y.tolist()==[[4,2,7]]
        x.fill_(0)
        y.fill_(0)
        np.testing.assert_array_equal(dataset,original)
        assert path.read_bytes()==original.tobytes()
    finally:
        del dataset


def test_output_device_can_differ_from_cpu_generator():
    x,y=get_batch(np.arange(8),2,3,"meta",generator=torch.Generator().manual_seed(42))
    assert x.device==y.device==torch.device("meta")
    assert x.shape==y.shape==(2,3)
    assert x.dtype==y.dtype==torch.int64


def test_invalid_device_is_not_silently_replaced_by_cpu():
    with pytest.raises((RuntimeError,ValueError)):
        get_batch(np.arange(8),2,3,"not-a-device")


def test_generator_state_can_resume_sampling_without_touching_global_rng():
    dataset=np.arange(1000)
    global_state=torch.get_rng_state().clone()
    generator=torch.Generator().manual_seed(42)
    get_batch(dataset,8,4,"cpu",generator=generator)
    saved=generator.get_state().clone()
    expected=get_batch(dataset,8,4,"cpu",generator=generator)
    assert not torch.equal(generator.get_state(),saved)
    resumed=torch.Generator().set_state(saved)
    actual=get_batch(dataset,8,4,"cpu",generator=resumed)
    for left,right in zip(actual,expected,strict=True):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),global_state)


def test_validation_generator_does_not_change_training_samples():
    dataset=np.arange(1000)
    train=torch.Generator().manual_seed(42)
    reference=torch.Generator().manual_seed(42)
    validation=torch.Generator().manual_seed(43)
    for _ in range(3):
        get_batch(dataset,8,4,"cpu",generator=validation)
        actual=get_batch(dataset,8,4,"cpu",generator=train)
        expected=get_batch(dataset,8,4,"cpu",generator=reference)
        for left,right in zip(actual,expected,strict=True):
            torch.testing.assert_close(left,right,rtol=0,atol=0)


def test_omitted_generator_uses_global_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        before=torch.get_rng_state().clone()
        actual=get_batch(np.arange(1000),8,4,"cpu")
        assert not torch.equal(torch.get_rng_state(),before)
        torch.manual_seed(42)
        expected=get_batch(np.arange(1000),8,4,"cpu")
        for left,right in zip(actual,expected,strict=True):
            torch.testing.assert_close(left,right,rtol=0,atol=0)


@pytest.mark.parametrize("dataset",[[1,2,3],None,np.ma.array([1,2,3],mask=[False,True,False])])
def test_invalid_dataset_type(dataset):
    with pytest.raises(TypeError,match="dataset"):
        get_batch(dataset,1,2,"cpu")


@pytest.mark.parametrize("dataset",[np.array(1),np.zeros((2,3),dtype=np.int64)])
def test_invalid_dataset_dimensions(dataset):
    with pytest.raises(ValueError,match="one-dimensional"):
        get_batch(dataset,1,2,"cpu")


@pytest.mark.parametrize("dtype",[np.float32,np.float64,np.bool_,np.complex64,object,"U2"])
def test_noninteger_dtypes_are_rejected(dtype):
    with pytest.raises(TypeError,match="integer dtype"):
        get_batch(np.array([0,1,1],dtype=dtype),1,2,"cpu")


@pytest.mark.parametrize("name",["batch_size","context_length"])
@pytest.mark.parametrize("value",[True,1.5,"2",None])
def test_invalid_size_types(name,value):
    args=dict(dataset=np.arange(8),batch_size=1,context_length=2,device="cpu")
    args[name]=value
    with pytest.raises(TypeError,match=name):
        get_batch(**args)


@pytest.mark.parametrize("name",["batch_size","context_length"])
@pytest.mark.parametrize("value",[0,-1])
def test_nonpositive_sizes(name,value):
    args=dict(dataset=np.arange(8),batch_size=1,context_length=2,device="cpu")
    args[name]=value
    with pytest.raises(ValueError,match=name):
        get_batch(**args)


@pytest.mark.parametrize("length",[0,1,3])
def test_insufficient_tokens(length):
    with pytest.raises(ValueError,match="context_length"):
        get_batch(np.arange(length),1,3,"cpu")


@pytest.mark.parametrize("value",[42,True,"cpu"])
def test_invalid_generator(value):
    with pytest.raises(TypeError,match="generator"):
        get_batch(np.arange(8),1,2,"cpu",generator=value)


@pytest.mark.parametrize("position",[0,1,2])
@pytest.mark.parametrize("value",[-1,2**63,2**64-1])
def test_sampled_token_out_of_int64_range(position,value):
    dataset=np.zeros(3,dtype=np.int64 if value<0 else np.uint64)
    dataset[position]=value
    with pytest.raises(ValueError,match="sampled token IDs"):
        get_batch(dataset,1,2,"cpu")


def test_unsampled_values_are_not_scanned():
    seed=42
    start=torch.randint(0,98,(1,),generator=torch.Generator().manual_seed(seed)).item()
    dataset=np.full(100,-1,dtype=np.int64)
    dataset[start:start+3]=[2,4,6]
    x,y=get_batch(dataset,1,2,"cpu",generator=torch.Generator().manual_seed(seed))
    assert x.tolist()==[[2,4]]
    assert y.tolist()==[[4,6]]
