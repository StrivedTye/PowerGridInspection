import torch
from .base_task import BaseTask

class SegTask(BaseTask):

    def __init__(self, cfg, log):
        super().__init__(cfg, log)

    def compute_loss(self, end_points, cfg):
        logits = end_points['logits']
        labels = end_points['labels']

        logits = logits.transpose(1, 2).reshape(-1, cfg.model_cfg.num_classes)
        labels = labels.reshape(-1)

        # Boolean mask of points that should be ignored
        ignored_bool = (labels == 0)
        for ign_label in cfg.model_cfg.ignored_labels:
            ignored_bool = ignored_bool | (labels == ign_label)

        # Collect logits and labels that are not ignored
        valid_idx = ignored_bool == 0
        valid_logits = logits[valid_idx, :]
        valid_labels_init = labels[valid_idx]

        # Reduce label values in the range of logit shape
        reducing_list = torch.arange(0, cfg.model_cfg.num_classes).long().cuda()
        inserted_value = torch.zeros((1,)).long().cuda()
        for ign_label in cfg.model_cfg.ignored_labels:
            reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
        valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
        loss = self.get_loss(valid_logits, valid_labels)
        end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
        end_points['loss'] = loss
        return loss, end_points

    def get_loss(self, logits, labels):
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        output_loss = criterion(logits, labels)
        return output_loss

    def training_step(self, batch, batch_idx):

        # Forward pass
        end_points = self.model(batch)

        if self.cfg.model_cfg.model_type == "KPConv":
            loss = self.model.loss(end_points['logits'], end_points['labels'])
        else:
            loss, _ = self.compute_loss(end_points, self.cfg)

        self.logger.experiment.add_scalars(
            'loss',
            {
                'loss_total': loss,
                'loss_ce': loss,
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