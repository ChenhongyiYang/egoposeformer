# Author: Chenhonghyi Yang
# Reference: https://github.com/hiroyasuakada/UnrealEgo

import os
import pickle
import random

from abc import ABCMeta

import numpy as np
import torch
from torch.utils.data import Dataset



class Unrealego3DPoseDataset(Dataset, metaclass=ABCMeta):
    _action_id_to_word = {
        1: "jumping",
        2: "falling_down",
        3: "exercising",
        4: "pulling",
        5: "singing",
        6: "rolling",
        7: "crawling",
        8: "laying",
        9: "sitting_on_the_ground",
        10: "crouching",
        11: "crouching_and_tuning",
        12: "crouching_to_standing",
        13: "crouching_and_moving_forward",
        14: "crouching_and_moving_backward",
        15: "crouching_and_moving_sideways",
        16: "standing_with_whole_body_movement",
        17: "standing_with_upper_body_movement",
        18: "standing_and_turning",
        19: "standing_to_crouching",
        20: "standing_and_moving_forward",
        21: "standing_and_moving_backward",
        22: "standing_and_moving_sideways",
        23: "dancing",
        24: "boxing",
        25: "wrestling",
        26: "soccer",
        27: "baseball",
        28: "basketball",
        29: "american_football",
        30: "golf",
    }

    def __init__(
        self,
        data_root,
        meta_path,
        info_json,
        action_id=0,
        pre_shuffle=False,
        **kwargs
    ):
        super(Unrealego3DPoseDataset, self).__init__()

        self.data_root = data_root
        self.meta_path = meta_path
        self.info_json = info_json
        self.action_id = int(action_id)

        with open(meta_path, 'rb') as f:
            pelvis_pos = pickle.load(f)
        self.pelvis_pos = pelvis_pos

        self.frame_dataset = self.collect_dataset(info_json, pre_shuffle)


    def collect_dataset(self, info_json, pre_shuffle):
        data = []

        with open(info_json, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            sub_path = line[16:]
            if self.action_id != 0:
                sub_action_id = int(sub_path.split("/")[-2])
                if sub_action_id != self.action_id:
                    continue

            frame_list = os.listdir(
                os.path.join(
                    self.data_root,
                    sub_path,
                    "all_data_with_img-256_hm-64_pose-16_npy"
                )
            )
            for i in range(len(frame_list)):
                data.append((sub_path, i))
        if pre_shuffle:
            random.shuffle(data)
        return data

    def load_data(self, idx):
        prefix, frame_idx = self.frame_dataset[idx]

        frame_path = os.path.join(
            self.data_root,
            prefix,
            "all_data_with_img-256_hm-64_pose-16_npy",
            "frame_%d.npy"%(frame_idx)
        )

        # load images
        frame_data = np.load(frame_path, allow_pickle=True)
        frame_data = frame_data.item()

        img_left = frame_data["input_rgb_left"][np.newaxis, :, :, :]  # [1, 3, 256, 256]
        img_right = frame_data["input_rgb_right"][np.newaxis, :, :, :]  # [1, 3, 256, 256]
        img = np.concatenate((img_left, img_right), axis=0)  # [2, 3, 256, 256]

        pose_gt = frame_data["gt_local_pose"][:, :]  # [16, 3]

        # load Pelvis position (for 3d to 2d projection)
        frame_key = str(os.path.join(prefix, "frame_%d"%(frame_idx)))

        pelvis_3d_left = self.pelvis_pos[frame_key][0]
        pelvis_3d_right = self.pelvis_pos[frame_key][1]

        pelvis_3d = np.array([pelvis_3d_left, pelvis_3d_right])
        pelvis_3d = pelvis_3d.reshape(2, 1, 3)

        ret_data = dict()
        ret_data["img"] = torch.from_numpy(img)
        ret_data["origin_3d"] = torch.from_numpy(pelvis_3d)
        ret_data["dataset_idx"] = idx
        ret_data["frame_path"] = str(frame_path)
        ret_data["gt_pose"] = torch.from_numpy(pose_gt)
        return ret_data

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx):
        return self.load_data(idx)
