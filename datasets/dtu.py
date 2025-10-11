from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2
import random
from torchvision import transforms


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, robust_train=False):
        super(MVSDataset, self).__init__()

        self.levels = 2
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        if self.mode == 'test':
            self.img_wh = (1600, 1152)  # (1600, 1200)
        else:
            self.img_wh = (640, 512)
        self.robust_train = robust_train

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        self.color_augment = transforms.ColorJitter(brightness=0.5, contrast=0.5)

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            pair_file = "Cameras/pair.txt"

            with open(os.path.join(self.datapath, pair_file)) as f:
                self.num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(self.num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    light_idx = random.randint(0, 6)
                    metas.append((scan, light_idx, ref_view, src_views))
                    # for light_idx in range(7):
                    #     metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_img(self, filename):
        img = Image.open(filename)
        if self.mode == 'train':
            img = self.color_augment(img)
        # scale 0~255 to -1~1
        np_img = 2 * np.array(img, dtype=np.float32) / 255. - 1
        h, w, _ = np_img.shape

        # new_h, new_w = (h // 56) * 56, (w // 56) * 56
        new_h, new_w = (h // 112) * 112, (w // 112) * 112
        np_img_ms = {
            "level_2": cv2.resize(np_img, (new_w // 4, new_h // 4), interpolation=cv2.INTER_LINEAR),
            "level_1": cv2.resize(np_img, (new_w // 2, new_h // 2), interpolation=cv2.INTER_LINEAR),
            "level_0": cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        }
        return np_img_ms

    def prepare_img(self, hr_img):
        # downsample
        h, w = hr_img.shape
        # new_h, new_w = (h // 56) * 56, (w // 56) * 56
        new_h, new_w = (h // 112) * 112, (w // 112) * 112
        hr_img = cv2.resize(hr_img, (new_w // 2, new_h // 2), interpolation=cv2.INTER_NEAREST)
        # crop
        h, w = hr_img.shape
        target_h, target_w = self.img_wh[1], self.img_wh[0]
        start_h, start_w = (new_h - target_h) // 2, (new_w - target_w) // 2
        hr_img_crop = hr_img[start_h: start_h + target_h, start_w: start_w + target_w]

        return hr_img_crop

    def read_mask(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        return np_img

    def read_depth_mask(self, filename, mask_filename, scale):
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32) * scale
        # depth_hr = np.squeeze(depth_hr, 2)
        depth_lr = self.prepare_img(depth_hr)
        mask = self.read_mask(mask_filename)
        mask = self.prepare_img(mask)
        mask = mask.astype(np.bool_)
        mask = mask.astype(np.float32)

        h, w = depth_lr.shape
        depth_lr_ms = {}
        mask_ms = {}

        for i in range(self.levels):
            depth_cur = cv2.resize(depth_lr, (w // (2 ** i), h // (2 ** i)), interpolation=cv2.INTER_NEAREST)
            mask_cur = cv2.resize(mask, (w // (2 ** i), h // (2 ** i)), interpolation=cv2.INTER_NEAREST)
            depth_lr_ms[f"level_{i}"] = depth_cur
            mask_ms[f"level_{i}"] = mask_cur

        return depth_lr_ms, mask_ms

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # robust training strategy
        if self.robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]
        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs_0 = []
        imgs_1 = []
        imgs_2 = []

        # mask = None

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, 'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            # mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            # depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))

            imgs = self.read_img(img_filename)
            imgs_0.append(imgs['level_0'])
            imgs_1.append(imgs['level_1'])
            imgs_2.append(imgs['level_2'])

            # if i == 0:  # reference view
            #     _, mask = self.read_depth_mask(depth_filename, mask_filename, 1)
            #
            #     for l in range(self.levels):
            #         mask[f'level_{l}'] = np.expand_dims(mask[f'level_{l}'], 2)
            #         mask[f'level_{l}'] = mask[f'level_{l}'].transpose([2, 0, 1])

        # imgs: N*3*H0*W0, N is number of images
        imgs_0 = np.stack(imgs_0).transpose([0, 3, 1, 2])
        imgs_1 = np.stack(imgs_1).transpose([0, 3, 1, 2])
        imgs_2 = np.stack(imgs_2).transpose([0, 3, 1, 2])

        imgs = {}
        imgs['level_0'] = imgs_0
        imgs['level_1'] = imgs_1
        imgs['level_2'] = imgs_2

        # data is numpy array
        return {"imgs": imgs,  # [N, 3, H, W]
                # "mask": mask    # [1, H, W]
                }
