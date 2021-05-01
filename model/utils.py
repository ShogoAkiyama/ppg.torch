import warnings
from functools import partial
import torch.distributions as dis
from gym3.types import Discrete, TensorType
import torch.nn as nn
import torch as th

node_types = {}


def tree_map(f, tree):
    """Map a function over a pytree to produce a new pytree.

  Args:
    f: function to be applied at each leaf.
    tree: a pytree to be mapped over.

  Returns:
    A new pytree with the same structure as `tree` but with the value at each
    leaf given by `f(x)` where `x` is the value at the corresponding leaf in
    `tree`.
  """
    node_type = node_types.get(type(tree))
    if node_type:
        children, node_spec = node_type.to_iterable(tree)
        new_children = [tree_map(f, child) for child in children]
        return node_type.from_iterable(node_spec, new_children)
    else:
        return f(tree)


def sum_nonbatch(x, nbatchdim=2):
    return x.sum(dim=tuple(range(
        nbatchdim, x.dim()))) if x.dim() > nbatchdim else x


def _make_categorical(x, ncat, shape):
    x = x.reshape((*x.shape[:-1], *shape, ncat))
    return dis.Categorical(logits=x)


def _make_normal(x, shape):
    warnings.warn("Using stdev=1")
    return dis.Normal(loc=x.reshape(x.shape[:-1] + shape), scale=1.0)


def _make_bernoulli(x, shape):  # pylint: disable=unused-argument
    return dis.Bernoulli(logits=x)


def tensor_distr_builder(ac_space):
    """
    Like distr_builder, but where ac_space is a TensorType
    """
    assert isinstance(ac_space, TensorType)
    eltype = ac_space.eltype
    if eltype == Discrete(2):
        return (ac_space.size, partial(_make_bernoulli, shape=ac_space.shape))
    if isinstance(eltype, Discrete):
        return (
            eltype.n * ac_space.size,
            partial(_make_categorical, shape=ac_space.shape, ncat=eltype.n),
        )
    else:
        raise ValueError(f"Expected ScalarType, got {type(ac_space)}")


def parse_dtype(x):
    if isinstance(x, th.dtype):
        return x
    elif isinstance(x, str):
        if x == "float32" or x == "float":
            return th.float32
        elif x == "float64" or x == "double":
            return th.float64
        elif x == "float16" or x == "half":
            return th.float16
        elif x == "uint8":
            return th.uint8
        elif x == "int8":
            return th.int8
        elif x == "int16" or x == "short":
            return th.int16
        elif x == "int32" or x == "int":
            return th.int32
        elif x == "int64" or x == "long":
            return th.int64
        elif x == "bool":
            return th.bool
        else:
            raise ValueError(f"cannot parse {x} as a dtype")
    else:
        raise TypeError(f"cannot parse {type(x)} as dtype")


def NormedConv2d(*args, scale=1, **kwargs):
    """
    nn.Conv2d but with normalized fan-in init
    """
    out = nn.Conv2d(*args, **kwargs)
    out.weight.data *= scale / out.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out


def NormedLinear(*args, scale=1.0, dtype=th.float32, **kwargs):
    """
    nn.Linear but with normalized fan-in init
    """
    dtype = parse_dtype(dtype)
    if dtype == th.float32:
        out = nn.Linear(*args, **kwargs)
    else:
        raise ValueError(dtype)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out


def intprod(xs):
    """
    Product of a sequence of integers
    """
    out = 1
    for x in xs:
        out *= x
    return out


def flatten_image(x):
    """
    Flattens last three dims
    """
    *batch_shape, h, w, c = x.shape
    return x.reshape((*batch_shape, h * w * c))


def sequential(layers, x, *args, diag_name=None):
    for (i, layer) in enumerate(layers):
        x = layer(x, *args)
    return x


def flatten_tensors(xs, dtype=None, buf=None):
    if buf is None:
        buf = xs[0].new_empty(sum(x.numel() for x in xs), dtype=dtype)
    i = 0
    for x in xs:
        buf[i : i + x.numel()].copy_(x.view(-1))
        i += x.numel()
    return buf


def transpose(x, before, after):
    """
    Usage: x_bca = transpose(x_abc, 'abc', 'bca')
    """
    assert sorted(before) == sorted(after), f"cannot transpose {before} to {after}"
    assert x.ndim == len(
        before
    ), f"before spec '{before}' has length {len(before)} but x has {x.ndim} dimensions: {tuple(x.shape)}"
    return x.permute(tuple(before.index(i) for i in after))
