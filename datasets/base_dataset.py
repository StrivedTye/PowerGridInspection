import torch
from abc import ABC, abstractmethod


class BaseDataset(ABC):

    def __init__(self, split_type, cfg, log):
        super().__init__()
        assert split_type in ['train', 'val', 'test']
        self.split_type = split_type
        self.cfg = cfg
        self.log = log

    @abstractmethod
    def get_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def num_samples(self):
        raise NotImplementedError

    @abstractmethod
    def get_sample(self, sample_id):
        raise NotImplementedError


class EvalDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: BaseDataset, cfg, log):
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg
        self.log = log

    def __len__(self):
        return self.dataset.num_samples()

    def __getitem__(self, idx):
        # read '*.las' file
        # pc = self.read_las(path[idx])
        pcd, label = self.dataset.get_sample(idx)

        # after getting sample, we need to do the following processes, as the range of 'pc' is very large and noisy.

        # step 1: preprocess: de-noise; down-sample

        # step 2: divide it into blocks


        # note that KPConv, DGCNN and RandLANet may have different inputs.
        # for DGCNN and RandLANet
        data = dict(pcds=pcd.astype('float32'),  # (n, 3)
                    labels=label.astype('int'),  # (n,)
                    )
        return data