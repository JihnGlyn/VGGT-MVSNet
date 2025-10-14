from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2
import random


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, nviews, img_wh):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.nviews = nviews
        self.img_wh = img_wh
        # self.img_wh = (1600, 1200)
        self.metas = self.build_list()

    def build_list(self):
        folder_metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            pair_file = "pair.txt"
            metas = []
            with open(os.path.join(self.datapath, scan, pair_file)) as f:
                self.num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(self.num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    metas.append((scan, ref_view, src_views[:self.nviews]))
                    # for light_idx in range(7):
                    #     metas.append((scan, light_idx, ref_view, src_views))
            folder_metas.append(metas)
            print("dataset", len(metas))
        print("folder_meta:", len(folder_metas))
        return folder_metas

    def __len__(self):
        return len(self.metas)

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to -1~1
        np_img = 2 * np.array(img, dtype=np.float32) / 255. - 1
        np_img = cv2.resize(np_img, self.img_wh, interpolation=cv2.INTER_LINEAR)
        h, w, _ = np_img.shape
        h, w = (h // 56) * 56, (w // 56) * 56
        np_img_ms = {
            "level_2": cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR),
            "level_1": cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR),
            "level_0": cv2.resize(np_img, (w, h), interpolation=cv2.INTER_LINEAR)
        }
        return np_img_ms

    def __getitem__(self, idx):
        # idx: 0-21
        pair_views = []
        folder_meta = self.metas[idx]
        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        for fid, metas in enumerate(folder_meta):
            # fid: 0-48
            scan, ref_view, src_views = metas
            pair_views.append(src_views)
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, fid))

            imgs = self.read_img(img_filename)
            imgs_0.append(imgs['level_0'])
            imgs_1.append(imgs['level_1'])
            imgs_2.append(imgs['level_2'])

        # imgs: N*3*H0*W0, N is number of images
        imgs_0 = np.stack(imgs_0).transpose([0, 3, 1, 2])
        imgs_1 = np.stack(imgs_1).transpose([0, 3, 1, 2])
        imgs_2 = np.stack(imgs_2).transpose([0, 3, 1, 2])

        imgs = {'level_0': imgs_0, 'level_1': imgs_1, 'level_2': imgs_2}

        pair_views = np.stack(pair_views)

        # data is numpy array
        return {
                "imgs": imgs,   # [B, 49, 3, H, W]
                "pair_view": pair_views,    # [B, 49, N]
                "filename": os.path.join(scan, "{}", "{:0>8}" + "{}"),
                "scan": os.path.join(scan),
        }
