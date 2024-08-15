# Author: Chenhongyi Yang


import os
from typing import Optional

import torch
from torch.utils.data import Dataset
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import ParallelStrategy

from torch.optim.lr_scheduler import MultiStepLR

from pose_estimation.models.estimator import EgoPoseFormerHeatmap
from pose_estimation.datasets import UnrealegoHeatmapDataset

def get_dataset(dataset_type, root, split, **kwargs):
    assert split in ["train", "test", "validation"]
    assert dataset_type in ["unrealego"]

    if dataset_type == "unrealego":
        if split == "train":
            return UnrealegoHeatmapDataset(
                data_root=os.path.join(root, "unrealego_impl"),
                info_json=os.path.join(root, "train.txt"),
                **kwargs
            )
        elif split == "validation":
            return UnrealegoHeatmapDataset(
                data_root=os.path.join(root, "unrealego_impl"),
                info_json=os.path.join(root, "validation.txt"),
                **kwargs
            )
        else:
            return UnrealegoHeatmapDataset(
                data_root=os.path.join(root, "unrealego_impl"),
                info_json=os.path.join(root, "test.txt"),
                **kwargs
            )
    else:
        raise NotImplementedError


class PoseHeatmapLightningModel(LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        dataset_type: str,
        data_root: str,
        lr: float,
        weight_decay: float,
        lr_decay_epochs: tuple,
        warmup_iters: int,
        batch_size: int,
        workers: int,
        dataset_kwargs: dict = {}
    ):
        super().__init__()

        assert dataset_type in ["unrealego"]
        self.dataset_type = dataset_type
        self.dataset_kwargs = dataset_kwargs

        self.model = EgoPoseFormerHeatmap(**model_cfg)

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epochs = lr_decay_epochs
        self.warmup_iters = warmup_iters

        self.data_root = data_root
        self.batch_size = batch_size
        self.workers = workers

        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        assert self.model.training

        image = batch["img"]
        heatmap_gt = batch["heatmap_gt"]
        loss_dict = self.model(image, heatmap_gt)
        loss_total = loss_dict.get("heatmap_loss")
        self.log("train_loss", loss_total)
        return loss_total

    def eval_step(self, batch, batch_idx, prefix):
        assert not self.model.training

        image = batch["img"]
        heatmap_gt = batch["heatmap_gt"]
        output_dict = self.model(image, heatmap_gt)
        self.log(f"{prefix}_l1_error", output_dict.get("l1_error").mean())
        self.log(f"{prefix}_pos_l1_error", output_dict.get("pos_l1_error").mean())
        return None

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure,):
        optimizer.step(closure=optimizer_closure)
        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.warmup_iters:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.warmup_iters))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = MultiStepLR(optimizer, self.lr_decay_epochs, gamma=0.1)
        return [optimizer], [scheduler]

    def setup(self, stage: str):
        if isinstance(self.trainer.strategy, ParallelStrategy):
            num_processes = max(1, self.trainer.strategy.num_processes)
            self.batch_size = int(self.batch_size / num_processes)
            self.workers = int(self.workers / num_processes)

        if stage == "fit":
            self.train_dataset = get_dataset(self.dataset_type, self.data_root, "train", **self.dataset_kwargs)

        if stage == "test" or stage == "predict":
            self.eval_dataset = get_dataset(self.dataset_type, self.data_root, "test", **self.dataset_kwargs)
        else:
            self.eval_dataset = get_dataset(self.dataset_type, self.data_root, "validation", **self.dataset_kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self):
        return self.val_dataloader()
