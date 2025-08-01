"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad == None:
              continue
            grad_data = ndl.Tensor(param.grad.data + self.weight_decay * param.data, dtype=param.dtype)
            if param not in self.u:
              self.u[param] = 0
            self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad_data 
            param.data = param.data - self.lr * self.u[param]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # print('2 global tensors', ndl.autograd.TENSOR_COUNTER)
        self.t += 1
        for param in self.params:
            deltaf = param.grad.data + self.weight_decay * param.data
            u_t = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * deltaf
            # u_t = ndl.Tensor(u_t, dtype=param.dtype)
            self.m[param] = u_t
            v_t = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * (deltaf ** 2)
            # v_t = ndl.Tensor(v_t, dtype=param.dtype)
            self.v[param] = v_t
            
            unbiased_u = self.m[param] / (1 - self.beta1 ** self.t)
            unbiased_v = self.v[param] / (1 - self.beta2 ** self.t)
            update = self.lr * unbiased_u.data / (unbiased_v.data ** 0.5 + self.eps)
            update = ndl.Tensor(update, dtype=param.dtype)
            # print(update)
            param.data -= update.data
        # print('3 global tensors', ndl.autograd.TENSOR_COUNTER)
        ### END YOUR SOLUTION
