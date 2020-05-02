import numpy as np
import cv2
import math
import scipy.io as sio
import h5py as H
import datasets.img as I
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import datasets.mpii_config as config


class MPII(Dataset):
    """docstring for MPII."""

    def __init__(self, split='train'):
        super(MPII, self).__init__()
        print("Starting loading {} images ... ...".format(split))
        self.split = split

        # worldCoods and raw_pts are same
        self.worldCoors = sio.loadmat(open(
            config.worldCoors[:-4] + ('train' if split is 'train' else '') + '.mat', 'rb'))['a']
        self.headsize = sio.loadmat(open(
            config.headSize[:-4] + ('train' if split is 'train' else '') + '.mat', 'rb'))['headSize']

        tags = ['imgname', 'part', 'center', 'scale']
        self.annot = self.__get_annotation(tags)
        self.len = len(self.annot['scale'])

        print("Successfully loaded {} {} images".format(self.len, split))

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        img = self.__load_image(i)
        pts, c, s = self.__get_info(i)
        r = 0

        # print(img.shape, pts)
        # plt.imshow(img)
        # plt.scatter(pts[:, 0], pts[:, 1], c='r', s=10)
        # plt.show()

        if self.split == 'train':
            s = s * (2 ** I.Rnd(config.max_scale))
            r = 0 if np.random.random() < 0.6 else I.Rnd(config.max_rotate)

        inp = I.Crop(img, c, s, r, config.input_res) / 256.

        out = np.zeros((config.n_joints, config.output_res, config.output_res))

        # print(c, s, r, config.output_res)
        for j in range(config.n_joints):
            if pts[j][0] > 1:
                pts[j] = I.Transform(pts[j], c, s, r, config.output_res)
                out[j] = I.DrawGaussian(
                    out[j], pts[j], config.gauss, 0.5 if config.output_res == 32 else -1)
        # print(inp.shape, pts)
        # plt.imshow(inp[0])
        # plt.scatter(pts[:, 0], pts[:, 1], c='r', s=10)
        # plt.show()

        inp_without_clip = inp.copy()
        if self.split == 'train':
            if np.random.random() < 0.5:
                inp = I.Flip(inp)
                out = I.ShuffleLR(I.Flip(out))
                # pts[:, 0] = config.output_res / 4 - pts[:, 0]
                # should be this, not the previous line
                pts[:, 0] = config.output_res - pts[:, 0]

            inp_without_clip = inp.copy()
            inp[0] = np.clip(inp[0] * (np.random.random() * (0.4) + 0.6), 0, 1)
            inp[1] = np.clip(inp[1] * (np.random.random() * (0.4) + 0.6), 0, 1)
            inp[2] = np.clip(inp[2] * (np.random.random() * (0.4) + 0.6), 0, 1)

        # print(inp.shape, pts)
        # plt.imshow(inp[0])
        # plt.scatter(pts[:, 0], pts[:, 1], c='r', s=10)
        # plt.show()

        if config.target_type == 'direct':
            out = np.reshape((pts / config.output_res), -1)  # - .5
            # print(inp.shape, out)
            return torch.from_numpy(inp).float(), \
                torch.from_numpy(out).float(), \
                torch.from_numpy(self.worldCoors[i]).float(), \
                torch.from_numpy(self.headsize[i]).float(), \
                torch.from_numpy(inp_without_clip).float()

        elif config.target_type == 'heatmap':
            return torch.from_numpy(inp), torch.from_numpy(out), self.worldCoors[i], self.headsize[i]

    def get_raw(self, i):
        img = self.__load_image(i)
        pts, c, s = self.__get_info(i)
        return img, pts

    def __get_info(self, i):
        pts = self.annot['part'][i].copy()
        c = self.annot['center'][i].copy()
        s = self.annot['scale'][i]
        s = s * 200
        return pts, c, s

    def __load_image(self, i):
        path = config.image_dir + self.annot['imgname'][i].decode("utf-8")
        img = cv2.imread(path)
        return img

    def __get_annotation(self, tags):
        f = H.File(config.annot_dir + self.split + ".h5", 'r')
        annot = {}
        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()
        return annot

    def plot_back(self, img, pts):
        img = img.data.numpy()
        # print(img.shape)
        img = np.moveaxis(img, 0, 2)
        # print(img.shape)
        pts = pts.reshape(-1, config.n_joints, 2).squeeze(0)
        pts = (pts + .5) * config.output_res
        plt.imshow(img)
        plt.scatter(pts[:, 0], pts[:, 1], c='r', s=10)
        plt.show()

    def plot_back_images(self, images, pts_list, cols=3, color='r'):
        rows = math.ceil(len(images) / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
        k = 0
        for i in range(rows):
            for j in range(cols):
                img, pts = images[k], pts_list[k]
                img = img.data.numpy()
                img = np.moveaxis(img, 0, 2)
                pts = pts.reshape(-1, config.n_joints, 2).squeeze(0)
                # pts = (pts + .5) * config.output_res
                pts = pts * config.output_res
                axs[i, j].imshow(img)
                axs[i, j].scatter(pts[:, 0], pts[:, 1], c=color, s=10)
                k += 1
        plt.show()
        # plt.savefig("output_images/computed_img_pts.jpg")
