import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # main_path = nn.Sequential(nn.Linear(in_features=dim, out_features=hidden_dim),
    #                           norm(dim=hidden_dim),
    #                           nn.ReLU(),
    #                           nn.Dropout(p=drop_prob),
    #                           nn.Linear(in_features=hidden_dim, out_features=dim),
    #                           norm(dim=dim))
    # result = nn.Residual(main_path)
    # return nn.Sequential(result, nn.ReLU())
    main_path = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(),
                              nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    res = nn.Residual(main_path)
    return nn.Sequential(res, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    path = nn.Sequential(nn.Linear(in_features=dim, out_features=hidden_dim),
                         nn.ReLU())
    res = []
    for block in range(num_blocks):
        res.append(ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob))
    return nn.Sequential(path, *res, nn.Linear(in_features=hidden_dim, out_features=num_classes))
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    total_loss, total_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    if opt:
        model.train()
        for X,y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            total_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            total_loss.append(loss.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    else:
        model.eval()
        for X,y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            total_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            total_loss.append(loss.numpy())
    sample_nums = len(dataloader.dataset)
    return total_error / sample_nums, np.mean(total_loss)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_set = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz",
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model = MLPResNet(28*28, hidden_dim=hidden_dim)

    train_set = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz",
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model = MLPResNet(28*28, hidden_dim=hidden_dim)
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in epochs:
        train_error, train_loss = epoch(train_loader, model, opt)
    test_error, test_loss = epoch(test_loader, model, None)
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
