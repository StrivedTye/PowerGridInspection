import torch
import pytorch_lightning as pl
import os.path as osp
import json
import time

from optimizers import create_optimizer
from schedulers import create_scheduler
from models import create_model
from utils import *


class BaseTask(pl.LightningModule):

    def __init__(self, cfg, log):
        super().__init__()
        self.cfg = cfg
        self.txt_log = log
        self.model = create_model(cfg.model_cfg, log)
        self.txt_log.info('Model size = %.2f MB' % self.compute_model_size())
        if 'KITTI' in cfg.dataset_cfg.dataset_type:
            self.iou = TorchIoU()
            self.acc = TorchAcc()
            self.runtime = TorchRuntime()
            if cfg.save_test_result:
                self.pred_bboxes = []

    def compute_model_size(self):
        num_param = sum([p.numel() for p in self.model.parameters()])
        param_size = num_param * 4 / 1024 / 1024  # MB
        return param_size

    def configure_optimizers(self):
        optimizer = create_optimizer(self.cfg.optimizer_cfg, self.parameters())
        scheduler = create_scheduler(self.cfg.scheduler_cfg, optimizer)
        return dict(
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

    def training_step(self, *args, **kwargs):
        raise NotImplementedError(
            'Training_step has not been implemented!')

    def on_validation_epoch_start(self):
        self.iou.reset()
        self.acc.reset()
        self.runtime.reset()

        # may split scene as its super large range
        # to do

    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        end_points = self.model(batch)
        end_time = time.time()
        runtime = end_time-start_time

        intersection, union, target = intersection_and_union_gpu(end_points['valid_logits'],
                                                                 end_points['valid_labels'],
                                                                 self.cfg.dataset_cfg.num_classes)
        self.iou(intersection, union)
        self.acc(intersection, target)
        self.runtime(torch.tensor(runtime, device=self.device))

    def on_validation_epoch_end(self):

        self.log('iou', self.iou.compute(), prog_bar=True)
        self.log('acc', self.acc.compute(), prog_bar=True)
        self.log('runtime', self.runtime.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        start_time = time.time()
        end_points = self.model(batch)
        end_time = time.time()
        runtime = end_time-start_time

        intersection, union, target = intersection_and_union_gpu(end_points['valid_logits'],
                                                                 end_points['valid_labels'],
                                                                 self.cfg.dataset_cfg.num_classes)
        self.iou(intersection, union)
        self.acc(intersection, target)
        self.runtime(torch.tensor(runtime, device=self.device))

    def on_test_epoch_start(self):
        self.iou.reset()
        self.acc.reset()
        self.runtime.reset()

        # may split scene as its super large range
        # to do

    def on_test_epoch_end(self):
        self.log('iou', self.iou.compute(), prog_bar=True)
        self.log('acc', self.acc.compute(), prog_bar=True)
        self.log('runtime', self.runtime.compute(), prog_bar=True)
        self.txt_log.info('Avg Prec/Succ=%.3f/%.3f Runtime=%.6f' % (
            self.iou.compute(), self.acc.compute(), self.runtime.compute()))
