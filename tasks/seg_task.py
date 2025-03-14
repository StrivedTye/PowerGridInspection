import torch.nn.functional as F
import torch
import time
import json
import os.path as osp

from datasets.utils.pcd_utils import *
from .base_task import BaseTask

class SegTask(BaseTask):

    def __init__(self, cfg, log):
        super().__init__(cfg, log)

    def compute_loss(self, end_points, cfg):
        logits = end_points['logits']
        labels = end_points['labels']

        logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
        labels = labels.reshape(-1)

        # Boolean mask of points that should be ignored
        ignored_bool = labels == 0
        for ign_label in cfg.ignored_label_inds:
            ignored_bool = ignored_bool | (labels == ign_label)

        # Collect logits and labels that are not ignored
        valid_idx = ignored_bool == 0
        valid_logits = logits[valid_idx, :]
        valid_labels_init = labels[valid_idx]

        # Reduce label values in the range of logit shape
        reducing_list = torch.range(0, cfg.num_classes).long().cuda()
        inserted_value = torch.zeros((1,)).long().cuda()
        for ign_label in cfg.ignored_label_inds:
            reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
        valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
        loss = self.get_loss(valid_logits, valid_labels, cfg.class_weights)
        end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
        end_points['loss'] = loss
        return loss, end_points

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = torch.from_numpy(pre_cal_weights).float().cuda()
        # one_hot_labels = F.one_hot(labels, self.config.num_classes)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        output_loss = criterion(logits, labels)
        output_loss = output_loss.mean()
        return output_loss

    def training_step(self, batch, batch_idx):

        # Forward pass
        end_points = self.model(batch)

        loss, _ = self.compute_loss(end_points, self.cfg)

        self.logger.experiment.add_scalars(
            'loss',
            {
                'loss_total': loss,
                'loss_bbox': loss,
                'loss_center': loss,
            },
            global_step=self.global_step
        )
        return loss

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            tensor_data[k] = torch.tensor(
                v, device=self.device, dtype=torch.float32).unsqueeze(0)
        return tensor_data