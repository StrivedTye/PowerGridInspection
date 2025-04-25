import torch
import numpy as np
import laspy as lp
from .base_dataset import BaseDataset, EvalDatasetWrapper


class PowerUAV(BaseDataset):
    def __init__(self, split_type, cfg, log):
        super().__init__(split_type, cfg, log)

        label_to_category = {
            0: "Noise",
            1: "Ground",
            2: "Low Vegetation",
            3: "Medium Vegetation",
            4: "Power line support tower",
            5: "Power lines",
        }

    def num_samples(self):
        return 1000

    def read_las(self, las_path):
        # with lp.open(las_path) as las:
        #     for points in spatial_chunk_iterator(las, chunk_size, 40):
        #         pcd = las_to_numpy(points, include_feats=['classification', 'rgb'])
        pass

    def get_sample(self, sample_id):

        # read '*.las' file
        # pcd = self.read_las(path[sample_id])

        # after loading pc, we need to crop samples.

        pcd = np.random.randn(3, self.cfg.num_points)
        label = np.concatenate([np.zeros(100),
                                np.ones(100),
                                np.ones(200) * 2,
                                np.ones(200) * 3,
                                np.ones(200) * 4,
                                np.ones(224) * 5,
                                ])
        return pcd, label

    def get_dataset(self):
        if self.split_type == 'train':
            return TrainDatasetWrapper(self, self.cfg, self.log)
        else:
            return EvalDatasetWrapper(self, self.cfg, self.log)


class TrainDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: BaseDataset, cfg, log):
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg
        self.log = log

    def __len__(self):
        return self.dataset.num_samples()

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            tensor_data[k] = torch.FloatTensor(v)
        return tensor_data

    def __getitem__(self, idx):
        pcd, label = self.dataset.get_sample(idx)

        # note that KPConv, DGCNN and RandLANet may have different inputs.
        # for DGCNN and RandLANet
        data = dict(pcds=pcd.astype('float32'),  # (3, n)
                    labels=label.astype('int'),  # (n,)
                    )
        return data

