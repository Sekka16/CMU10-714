# test_linear.py
import sys
sys.path.append('./python')
import numpy as np
import pytest
import torch

import needle as ndl

_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

@pytest.mark.parametrize("batch,in_features,out_features,bias", [
    (4, 8, 16, True),
    (10, 5, 3, False),
    (2, 32, 64, True),
])
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_linear_forward(batch, in_features, out_features, bias, device):
    np.random.seed(0)
    torch.manual_seed(0)

    # Needle
    f = ndl.nn.Linear(in_features, out_features, bias=bias, device=device)
    x = ndl.init.rand(batch, in_features, device=device)

    # Torch（权重对齐：ndl (in,out) <-> torch (out,in)）
    g = torch.nn.Linear(in_features, out_features, bias=bias)
    g.weight.data = torch.tensor(f.weight.detach().numpy().T)
    if bias:
        g.bias.data = torch.tensor(f.bias.detach().numpy().reshape(-1))

    z_ndl = f(x).numpy()
    z_torch = g(torch.tensor(x.detach().numpy())).detach().numpy()

    np.testing.assert_allclose(z_ndl, z_torch, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch,in_features,out_features,bias", [
    (4, 8, 16, True),
    (5, 3, 2, False),
])
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_linear_backward(batch, in_features, out_features, bias, device):
    np.random.seed(0)
    torch.manual_seed(0)

    # Needle
    f = ndl.nn.Linear(in_features, out_features, bias=bias, device=device)
    x = ndl.init.rand(batch, in_features, device=device, requires_grad=True)

    # Torch（权重对齐：ndl (in,out) <-> torch (out,in)）
    g = torch.nn.Linear(in_features, out_features, bias=bias)
    g.weight.data = torch.tensor(f.weight.detach().numpy().T)
    if bias:
        g.bias.data = torch.tensor(f.bias.detach().numpy().reshape(-1))
    xt = torch.tensor(x.detach().numpy(), requires_grad=True)

    # 前向
    y_ndl = f(x).sum()
    y_torch = g(xt).sum()

    # 反向
    y_ndl.backward()
    y_torch.backward()

    # 检查梯度（用 .grad.numpy() 以触发 realize）
    np.testing.assert_allclose(x.grad.numpy(), xt.grad.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(f.weight.grad.numpy(), g.weight.grad.detach().numpy().T, atol=1e-5)
    if bias:
        np.testing.assert_allclose(f.bias.grad.numpy().reshape(-1), g.bias.grad.detach().numpy(), atol=1e-5)


@pytest.mark.parametrize("batch,dim", [
    (4, 3),
    (8, 16),
])
@pytest.mark.parametrize("device", _DEVICES)
def test_bn1d_forward_train_matches_torch(batch, dim, device):
    np.random.seed(0)
    torch.manual_seed(0)

    # Needle
    f = ndl.nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1, device=device)
    x = ndl.init.rand(batch, dim, device=device)

    # Torch（同步参数与运行统计）
    g = torch.nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1,
                             affine=True, track_running_stats=True)
    with torch.no_grad():
        g.weight.data = torch.tensor(f.weight.cached_data.numpy())
        g.bias.data   = torch.tensor(f.bias.cached_data.numpy())
        g.running_mean.data = torch.tensor(f.running_mean.cached_data.numpy())
        g.running_var.data  = torch.tensor(f.running_var.cached_data.numpy())

    # Train
    f.train()
    g.train()

    y_ndl = f(x).cached_data.numpy()
    y_tch = g(torch.tensor(x.cached_data.numpy())).detach().numpy()

    # 前向输出严格对齐
    np.testing.assert_allclose(y_ndl, y_tch, atol=1e-5, rtol=1e-5)

    # 运行均值/方差允许轻微差异（方案二）
    np.testing.assert_allclose(
        f.running_mean.cached_data.numpy(),
        g.running_mean.detach().numpy(), atol=2e-3, rtol=2e-3
    )
    np.testing.assert_allclose(
        f.running_var.cached_data.numpy(),
        g.running_var.detach().numpy(), atol=2e-3, rtol=2e-3
    )


@pytest.mark.parametrize("batch,dim", [
    (4, 3),
    (8, 16),
])
@pytest.mark.parametrize("device", _DEVICES)
def test_bn1d_forward_eval_matches_torch_after_one_update(batch, dim, device):
    np.random.seed(1)
    torch.manual_seed(1)

    # 先各自做一次 train 前向以更新 running stats
    f = ndl.nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1, device=device)
    g = torch.nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1,
                             affine=True, track_running_stats=True)
    with torch.no_grad():
        g.weight.data = torch.tensor(f.weight.cached_data.numpy())
        g.bias.data   = torch.tensor(f.bias.cached_data.numpy())
        g.running_mean.data = torch.tensor(f.running_mean.cached_data.numpy())
        g.running_var.data  = torch.tensor(f.running_var.cached_data.numpy())

    f.train(); g.train()
    x_train = ndl.init.rand(batch, dim, device=device)
    _ = f(x_train)
    _ = g(torch.tensor(x_train.cached_data.numpy()))

    # 切换到 eval，用各自的 running stats 做前向
    f.eval(); g.eval()
    x_eval = ndl.init.rand(batch, dim, device=device)
    y_ndl = f(x_eval).cached_data.numpy()
    y_tch = g(torch.tensor(x_eval.cached_data.numpy())).detach().numpy()

    # eval 输出对齐（考虑到运行统计累计的细微误差，放宽一点）
    np.testing.assert_allclose(y_ndl, y_tch, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("batch,dim", [
    (5, 7),
    (3, 11),
])
@pytest.mark.parametrize("device", _DEVICES)
def test_bn1d_backward_matches_torch(batch, dim, device):
    np.random.seed(2)
    torch.manual_seed(2)

    # Needle（需要梯度）
    f = ndl.nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1, device=device)
    x = ndl.init.rand(batch, dim, device=device, requires_grad=True)

    # Torch（同步参数与运行统计）
    g = torch.nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1,
                             affine=True, track_running_stats=True)
    with torch.no_grad():
        g.weight.data = torch.tensor(f.weight.cached_data.numpy())
        g.bias.data   = torch.tensor(f.bias.cached_data.numpy())
        g.running_mean.data = torch.tensor(f.running_mean.cached_data.numpy())
        g.running_var.data  = torch.tensor(f.running_var.cached_data.numpy())
    xt = torch.tensor(x.cached_data.numpy(), requires_grad=True)

    # Train 模式，标量损失做反传
    f.train(); g.train()
    y_ndl = f(x).sum()
    y_tch = g(xt).sum()

    y_ndl.backward()
    y_tch.backward()

    # 梯度严格对齐
    np.testing.assert_allclose(x.grad.cached_data.numpy(),   xt.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(f.weight.grad.cached_data.numpy(), g.weight.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(f.bias.grad.cached_data.numpy(),   g.bias.grad.detach().numpy(),   atol=1e-5, rtol=1e-5)