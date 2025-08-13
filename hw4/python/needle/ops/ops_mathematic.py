"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * a.log()
        return grad_a, grad_b
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return self.scalar * out_grad * (node.inputs[0] ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad / b
        grad_b = -out_grad * a / (b * b)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dim = list(range(len(a.shape)))
        if self.axes is None:
            dim[-2], dim[-1] = dim[-1], dim[-2]
        else:
            dim[self.axes[0]], dim[self.axes[1]] = dim[self.axes[1]], dim[self.axes[0]]
        return a.permute(dim)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)

class Permute(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.permute(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.permute(node.inputs[0].axes)
        ### END YOUR SOLUTION


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        origin_shape = node.inputs[0].shape
        return out_grad.reshape(origin_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if a.shape == self.shape:
            return a
        return array_api.broadcast_to(a, self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        origin_shape = node.inputs[0].shape
        if origin_shape == self.shape:
            return out_grad

        shrink_dims = [i for i in range(len(self.shape))]
        # iterate from the back because it could be len(ori_shape) < len(self.shape)
        for i, (ori, cur) in enumerate(zip(reversed(origin_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        assert len(shrink_dims) > 0

        return out_grad.sum(shrink_dims).reshape(origin_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

# class Summation(TensorOp):
#     def __init__(self, axes: Optional[tuple] = None):
#         self.axes = axes

#     def compute(self, a):
#         ### BEGIN YOUR SOLUTION
#         return a.sum(self.axes)
#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         # 我们需要将维度扩张到原先的维度，再在对应的维度上做广播
#         # 举例来说：(3,4,5)->(3,5)，那么我们需要(3,5)->(3,1,5)->(3,4,5)
#         input_shape = node.inputs[0].shape
#         axes = self.axes
#         if axes is None:
#             reshape_shape = [1] * len(input_shape)
#         else:
#             if isinstance(axes, int):
#                 axes = (axes,)
#             reshape_shape = list(input_shape)
#             for ax in axes:
#                 reshape_shape[ax] = 1
#         return out_grad.reshape(tuple(reshape_shape)).broadcast_to(input_shape)
#         ### END YOUR SOLUTION


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis = axis)
            return a
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        input_shape = input.shape
        
        expand_dims = list(input_shape)             # 需要扩展到什么维度
        if self.axes is None:                       # 说明summation的结果是矩阵里所有值的和
            axes = list(range(len(input_shape)))
        else:                                       # 说明规定了在哪些维度上求和
            if isinstance(self.axes, int):
                axes = [self.axes]
            else:
                axes = self.axes
        for i in range(len(axes)):
            expand_dims[axes[i]] = 1
        out_grad = reshape(out_grad, expand_dims)   # 调整输出梯度的形状，恢复被压缩的维度（尺寸为1）
        return broadcast_to(out_grad, input_shape)  # 例如：(2,1,4)广播为(2,3,4)，与输入形状一致
        ### END YOUR SOLUTION

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad @ b.transpose()
        grad_b = a.transpose() @ out_grad
        # 使用广播机制，将梯度扩展到与输入张量相同的形状
        if a.shape != grad_a.shape:
            grad_a.sum(tuple(range(len(grad_a.shape) - len(a.shape))))
        if b.shape != grad_b.shape:
            grad_b.sum(tuple(range(len(grad_b.shape) - len(b.shape))))
        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.maximum(0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_data = node.inputs[0].realize_cached_data()
        return out_grad * (input_data.maximum(0) > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - node ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(args) > 0, "Stack needs at least one array!"
        shape = args[0].shape
        for a in args:
            assert shape == a.shape, "All arrays need to be of the same size!"
        n = len(args)
        new_shape = list(shape)
        new_shape.insert(self.axis, n)
        out = array_api.empty(new_shape, device=args[0].device)
        slices = [slice(0, s) for s in new_shape]
        for i, arr in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            out[tuple(slices)] = arr
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n = A.shape[self.axis] 
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        slices = [slice(0, s) for s in A.shape]
        splits = []
        for i in range(n):
            slices[self.axis] = slice(i, i+1)
            splits.append(A[tuple(slices)].compact().reshape(new_shape))
        return tuple(splits)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        slices = [slice(None)] * len(a.shape)
        for axis in self.axes:
            new_shape[axis] *= (self.dilation + 1)
            slices[axis] = slice(None, None, self.dilation + 1)
        
        out = array_api.full(new_shape, 0.0, dtype=a.dtype, device=a.device)
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = [slice(None)] * len(a.shape)
        for axis in self.axes:
            slices[axis] = slice(None, None, self.dilation + 1)
        out = a[tuple(slices)]
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # A: (N, H, W, C_in), B: (K, K, C_in, C_out)
        pad_axes = [(0, 0)] + [(self.padding, self.padding)] * (A.ndim - 2) + [(0, 0)]
        A = A.pad(pad_axes)

        N, H, W, C_in = A.shape
        Ns, Hs, Ws, Cs = A.strides
        K = B.shape[0]
        out_h = H - K + 1
        out_w = W - K + 1
        inner_dim = K * K * C_in

        A_strided = A.as_strided(
            shape=(N, out_h, out_w, K, K, C_in),
            strides=(Ns, Hs, Ws, Hs, Ws, Cs)
        )

        A_2d_shape = (N * out_h * out_w, inner_dim)
        A_2d = A_strided.compact().reshape(A_2d_shape)

        B_2d = B.compact().reshape((inner_dim, B.shape[3]))
        out = A_2d @ B_2d

        out_4d = out.reshape((N, out_h, out_w, B.shape[3]))

        if self.stride > 1:
            out_4d = out_4d[:, ::self.stride, ::self.stride, :]
        return out_4d
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.realize_cached_data().strides
        
        if self.stride > 1: # If the convolution is strided, increase the size of `out_grad` with a corresponding dilation
            out_grad = dilate(out_grad, (1, 2), dilation=self.stride - 1)
            
        # `X.grad = ≈conv(≈out_grad, ≈W)`
        B_t = flip(B, (0, 1)).transpose((2, 3)) # `W` should be flipped over both the kernel dimensions
        # This padding depends on both the kernel size and the `padding` argument to the convolution
        A_grad = conv(out_grad, B_t, padding = K - 1 - self.padding)
        
        # `W.grad = ≈conv(≈X, ≈out_grad)` W.shape = (K, K, C_in, C_out)
        # A.shape: (N, H, W ,C_in) --> (C_in, H, W, N)
        A_t = A.transpose((0, 3))   # Consider turning batches into channels via transpose/permute
        # out_grad dimensions: (N, H-K+1, W-K+1, C_out) --> (H-K+1, W-k+1, N, C_out)
        out_grad_t = out_grad.permute((1, 2, 0, 3))
        #  conv( (C_in, H, W, N),  (H-K+1, W-k+1, N, C_out) ) --> (C_in, K, K, C_out)
        B_grad_t = conv(A_t, out_grad_t, padding=self.padding)
        B_grad = B_grad_t.permute((1, 2, 0, 3))
        
        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


