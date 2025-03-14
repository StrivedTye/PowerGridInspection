import numpy as np
import torch
import torchmetrics.utilities.data
from torchmetrics import Metric


def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TorchIoU(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0.0, dtype=torch.float),
                       dist_reduce_fx='sum')
        self.add_state("union", default=torch.tensor(0.0, dtype=torch.float),
                       dist_reduce_fx='sum')

    def update(self, intersection, union):
        self.intersection += intersection
        self.union += union

    def compute(self):
        return self.intersection / (self.union + 1e-10)


class TorchAcc(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0.0, dtype=torch.float),
                       dist_reduce_fx='sum')
        self.add_state("target", default=torch.tensor(0.0, dtype=torch.float),
                       dist_reduce_fx='sum')

    def update(self, intersection, target):
        self.intersection += intersection
        self.target += target

    def compute(self):
        return self.intersection / (self.target + 1e-10)


class TorchRuntime(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("sum_runtime", default=torch.tensor(0.0, dtype=torch.float),
                       dist_reduce_fx='sum')

    def update(self, runtime, n_runs):
        self.sum_runtime += runtime

    def compute(self):
        return self.sum_runtime