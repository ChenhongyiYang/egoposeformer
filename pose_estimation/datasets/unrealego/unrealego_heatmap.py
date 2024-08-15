# Author: Chenhonghyi Yang
# Reference: https://github.com/hiroyasuakada/UnrealEgo

import os
from abc import ABCMeta

import numpy as np
import torch
from torch.utils.data import Dataset


class UnrealegoHeatmapDataset(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        data_root,
        info_json,
        **kwargs
    ):
        super(UnrealegoHeatmapDataset, self).__init__()

        self.data_root = data_root
        self.info_json = info_json
        self.preprocess_path = "all_data_with_img-256_hm-64_pose-16_npy"

        self.dataset = self.collect_dataset()

    def collect_dataset(self):
        data = []

        with open(self.info_json) as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if len(line) == 0:
                continue
            sub_path = line[16:]
            file_list = os.listdir(
                os.path.join(self.data_root, sub_path, self.preprocess_path)
            )
            for file_name in file_list:
                data.append(
                    (os.path.join(self.data_root, sub_path, self.preprocess_path, file_name), 0)
                )  # left
                data.append(
                    (os.path.join(self.data_root, sub_path, self.preprocess_path, file_name), 1)
                )  # right
        return data

    def load_data(self, idx):
        path, view = self.dataset[idx]

        data = np.load(path, allow_pickle=True)
        data = data.item()

        if view == 0:
            img = data["input_rgb_left"][np.newaxis, :, :, :]  # [1, 3, 256, 256]
            heatmap = data["gt_heatmap_left"][np.newaxis, :, :, :]  # [15, 64, 64]
        else:
            img = data["input_rgb_right"][np.newaxis, :, :, :]  # [3, 256, 256]
            heatmap = data["gt_heatmap_right"][np.newaxis, :, :, :]  # [15, 64, 64]

        ret_data = dict()
        ret_data["img"] = torch.from_numpy(img)
        ret_data["heatmap_gt"] = torch.from_numpy(heatmap)
        return ret_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.load_data(idx)

