import torch
from torch import Tensor

__all__ = ["_pwpa","pwpa","aos2soa"]


def _pwpa(x: Tensor, coeffs: Tensor, partition_points: Tensor, AoS: bool = True) -> Tensor:
    """Performs PwPA in an efficient fused kernel"""
    return torch.ops.torch_pace._pwpa.default(x, coeffs, partition_points, AoS)

def pwpa(x: Tensor, coeffs: Tensor, partition_points: Tensor, AoS: bool = True) -> Tensor:
    """Performs PwPA in an efficient fused kernel"""
    return torch.ops.torch_pace.pwpa.default(x, coeffs, partition_points, AoS)

def aos2soa(coeffs: Tensor, D: int) -> Tensor:
    """
    aos2soa transform an array of structures (AoS) in an structure of array (SoA).

    This method uses :meth:`at::im2col` (a.k.a. :meth:`torch.nn.Unfold`) in order
    to perform the transformation.

    Args:
        coeffs (Tensor): array to transform.
        P (int): size of each array in SoA.
    """
    return torch.ops.torch_pace.aos2soa.default(coeffs, D)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("torch_pace::_pwpa")
def _(a, b, c, d):
    torch._check(a.dtype == torch.float32 or a.dtype == torch.float16)
    torch._check(b.dtype == torch.float32 or b.dtype == torch.float16)
    torch._check(c.dtype == torch.float32 or c.dtype == torch.float16)
    torch._check(a.device == b.device)
    torch._check(a.device == c.device)
    return torch.empty_like(a)

# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("torch_pace::pwpa")
def _(a, b, c, d):
    torch._check(a.dtype == torch.float32 or a.dtype == torch.float16)
    torch._check(b.dtype == torch.float32 or b.dtype == torch.float16)
    torch._check(c.dtype == torch.float32 or c.dtype == torch.float16)
    torch._check(a.device == b.device)
    torch._check(a.device == c.device)
    return torch.empty_like(a)

# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("torch_pace::aos2soa")
def _(a, b):
    torch._check(a.dtype == torch.float32 or a.dtype == torch.float16)
    return torch.empty_like(a)



def _backward(ctx, grad):
    a = ctx.saved_tensors
    grad_a, grad_b = None, None
    # if ctx.needs_input_grad[0]:
    #     grad_a = torch.ops.torch_pace.mymul.default(grad, b)
    # if ctx.needs_input_grad[1]:
    #     grad_b = torch.ops.torch_pace.mymul.default(grad, a)
    return grad_a, grad_b, None


def _setup_context(ctx, inputs, output):
    a, b, c, d = inputs
    saved_a, saved_b, saved_c = None, None, None
    if ctx.needs_input_grad[0]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b, saved_c)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
# torch.library.register_autograd(
#     "torch_pace::_pwpa", _backward, setup_context=_setup_context)
# torch.library.register_autograd(
#     "torch_pace::pwpa", _backward, setup_context=_setup_context)
# torch.library.register_autograd(
#     "torch_pace::aos2soa", _backward, setup_context=_setup_context)

