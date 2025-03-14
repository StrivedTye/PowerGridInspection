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
    def get_sample(self, tracklet_id, frame_id):
        raise NotImplementedError


class EvalDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: BaseDataset, cfg, log):
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg
        self.log = log

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass