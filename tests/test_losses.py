"""Numerical regressions for the migrated loss; optional CPU compute tests."""

import math

import pytest

from tiny_lm.losses import cross_entropy

torch = pytest.importorskip("torch", reason="loss tests require the optional compute dependencies")
F = torch.nn.functional


@pytest.mark.parametrize("shape", [(7,), (3, 7), (2, 3, 7)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_loss_and_gradient_match_pytorch(shape, dtype):
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
    targets = torch.randint(shape[-1], shape[:-1], generator=generator)
    reference_logits = logits.detach().clone().requires_grad_()

    actual = cross_entropy(logits, targets)
    expected = F.cross_entropy(reference_logits.reshape(-1, shape[-1]), targets.reshape(-1))
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    expected_gradient = torch.autograd.grad(expected, reference_logits)[0]

    assert actual.shape == torch.Size([])
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_noncontiguous_inputs():
    logits = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4).transpose(0, 1)
    targets = torch.tensor([[0, 1, 2], [1, 2, 3]]).transpose(0, 1)
    assert not logits.is_contiguous()
    assert not targets.is_contiguous()
    torch.testing.assert_close(
        cross_entropy(logits, targets),
        F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1)),
    )


def test_uniform_large_logits_keep_small_loss():
    # The old add-max-then-subtract-target ordering returned 0 for this input.
    logits = torch.full((1, 4), 1e8, dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0])
    actual = cross_entropy(logits, targets)
    gradient = torch.autograd.grad(actual, logits)[0]

    assert actual.item() == pytest.approx(math.log(4), rel=1e-6)
    torch.testing.assert_close(gradient, torch.tensor([[-0.75, 0.25, 0.25, 0.25]]))


def test_common_offset_preserves_loss_and_gradient():
    # Integer differences remain exactly representable after this float32 shift,
    # so this isolates computation error from input quantization.
    original = torch.tensor([[0.0, 1.0, -2.0, 3.0], [-3.0, 2.0, 1.0, 0.0]])
    targets = torch.tensor([1, 3])
    base = original.clone().requires_grad_()
    shifted = (original + 1e6).requires_grad_()
    base_loss = cross_entropy(base, targets)
    shifted_loss = cross_entropy(shifted, targets)

    torch.testing.assert_close(shifted_loss, base_loss)
    torch.testing.assert_close(
        torch.autograd.grad(shifted_loss, shifted)[0],
        torch.autograd.grad(base_loss, base)[0],
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_use_float32_computation(dtype):
    logits = torch.tensor(
        [[3.125, -0.25, 1.5, 4.0], [0.5, 0.25, -2.0, 1.0]], dtype=dtype, requires_grad=True
    )
    targets = torch.tensor([1, 3])
    # Promote the same quantized input, not an independently generated tensor.
    reference_logits = logits.detach().double().requires_grad_()
    actual = cross_entropy(logits, targets)
    expected = F.cross_entropy(reference_logits, targets)
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    expected_gradient = torch.autograd.grad(expected, reference_logits)[0]

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)
    epsilon = torch.finfo(dtype).eps
    torch.testing.assert_close(
        actual_gradient, expected_gradient.to(dtype), rtol=epsilon, atol=epsilon / 16
    )


def test_float16_large_vocabulary_does_not_overflow():
    # exp(0) summed over this vocabulary overflowed float16 in the old loss.
    logits = torch.zeros((1, 100_000), dtype=torch.float16, requires_grad=True)
    actual = cross_entropy(logits, torch.tensor([0]))
    gradient = torch.autograd.grad(actual, logits)[0]

    assert actual.dtype == torch.float32
    assert actual.item() == pytest.approx(math.log(100_000), rel=1e-6)
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0].item() == pytest.approx(-1.0, abs=1e-3)
    torch.testing.assert_close(gradient[0, 1:], torch.full_like(gradient[0, 1:], 1 / 100_000))


def test_extreme_score_gaps_have_finite_loss_and_gradient():
    logits = torch.tensor([[10_000.0, -10_000.0, 0.0]], requires_grad=True)
    actual = cross_entropy(logits, torch.tensor([1]))
    gradient = torch.autograd.grad(actual, logits)[0]
    assert actual.item() == pytest.approx(20_000.0)
    torch.testing.assert_close(gradient, torch.tensor([[1.0, -1.0, 0.0]]))


def test_float64_gradient_matches_finite_differences():
    logits = torch.tensor([[0.2, -0.7, 1.3], [-1.0, 0.5, 0.1]], dtype=torch.float64)
    logits.requires_grad_()
    assert torch.autograd.gradcheck(cross_entropy, (logits, torch.tensor([2, 0])))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64])
def test_invalid_logits_dtype(dtype):
    with pytest.raises(TypeError, match="logits must have dtype"):
        cross_entropy(torch.zeros((2, 3), dtype=dtype), torch.tensor([0, 1]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_invalid_targets_dtype(dtype):
    with pytest.raises(TypeError, match="targets must have dtype int64"):
        cross_entropy(torch.zeros((2, 3)), torch.zeros(2, dtype=dtype))


@pytest.mark.parametrize(
    ("logits_shape", "targets_shape", "message"),
    [
        ((), (), "vocabulary axis"),
        ((2, 0), (2,), "vocabulary axis"),
        ((2, 3), (2, 1), "targets.shape"),
        ((2, 3), (1,), "targets.shape"),
        ((0, 3), (0,), "at least one target"),
    ],
)
def test_invalid_shapes(logits_shape, targets_shape, message):
    with pytest.raises(ValueError, match=message):
        cross_entropy(torch.zeros(logits_shape), torch.zeros(targets_shape, dtype=torch.int64))


@pytest.mark.parametrize("target", [-100, -1, 3])
def test_invalid_target_ids_are_rejected(target):
    with pytest.raises((RuntimeError, IndexError)):
        cross_entropy(torch.zeros((1, 3)), torch.tensor([target]))


def test_device_mismatch():
    with pytest.raises(ValueError, match="same device"):
        cross_entropy(torch.zeros((1, 3)), torch.zeros(1, dtype=torch.int64, device="meta"))


@pytest.mark.parametrize(
    ("logits", "targets"), [([[0.0, 1.0]], torch.tensor([0])), (torch.zeros((1, 2)), [0])]
)
def test_non_tensor_inputs(logits, targets):
    with pytest.raises(TypeError, match="torch.Tensor"):
        cross_entropy(logits, targets)
