import argparse
import operator
from functools import reduce
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_cuda_memory_snapshot(device: torch.device) -> Dict[str, float]:
    stats = {
        "peak_allocated": 0.0,
        "peak_reserved": 0.0,
    }

    if device.type != "cuda":
        return stats

    stats["peak_allocated"] = float(torch.cuda.max_memory_allocated(device=device))
    stats["peak_reserved"] = float(torch.cuda.max_memory_reserved(device=device))

    return stats


def get_pos_lst(size_lst, length):
    pos_lst = []
    if len(size_lst[0]) == 2:
        for size in size_lst:
            nx, ny = size
            pos_x = torch.linspace(0, length[0], steps=nx + 1).float().cuda()[:-1].unsqueeze(-1)
            pos_y = torch.linspace(0, length[1], ny).float().cuda().unsqueeze(-1)
            pos_lst.append([pos_x, pos_y])
    elif len(size_lst[0]) == 3:
        for size in size_lst:
            nx, ny, nz = size
            pos_x = torch.linspace(0, length[0], nx).float().cuda().unsqueeze(-1)
            pos_y = torch.linspace(0, length[1], ny).float().cuda().unsqueeze(-1)
            pos_z = torch.linspace(0, length[2], nz).float().cuda().unsqueeze(-1)
            pos_lst.append([pos_x, pos_y, pos_z])
    else:
        raise ValueError
    
    return pos_lst


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.shape[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
    
    
# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c


def plot_loss_curve(x_range, y_range, train_losses, test_losses, save_path):
    """
    绘制训练/测试损失曲线。

    参数:
    - x_range: (xmin, xmax)，横轴范围（epoch范围），例如 (1, total_epochs)
    - y_range: (ymin, ymax)，纵轴范围（log尺度），例如 (1e-3, 1.0)
    - train_losses: 训练损失序列 (list 或 1D numpy/tensor)
    - test_losses: 测试损失序列 (list 或 1D numpy/tensor)
    - save_path: 图片保存路径
    """
    # 安全转换为 numpy 数组
    train_losses = np.asarray(train_losses, dtype=np.float32)
    test_losses = np.asarray(test_losses, dtype=np.float32)

    # 构造 x 轴（epoch 序号从 1 开始）
    x_train = np.arange(1, len(train_losses) + 1, dtype=np.int32)
    x_test = np.arange(1, len(test_losses) + 1, dtype=np.int32)

    # 创建图像
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('Training vs Test Loss (log scale)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.4)

    # 画曲线
    if len(train_losses) > 0:
        ax.plot(x_train, train_losses, label='Train', color='tab:blue', linewidth=2)
    if len(test_losses) > 0:
        ax.plot(x_test, test_losses, label='Test', color='tab:orange', linewidth=2)
    ax.legend()

    # 保存并覆盖前一张图
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    length = [3.0, 1.0]
    size_lst = [(72,24)]
    pos_lst = get_pos_lst(size_lst, length)[0]
