import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random


class Problem4Solver(object):

    def __init__(self):
        self.sample_img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                    ])
        # self.img = self.sample_img

    def solve(self, img, img_name):
        self.img = img
        rows, cols = self.img.shape
        f1 = np.zeros(self.img.shape)
        f2 = np.zeros(self.img.shape)
        for y in range(rows):
            for x in range(cols):
                if self.img[y, x] > 0:
                    f1[y, x] = f1[y, x - 1] + 1
                elif self.img[y, x] == 0:
                    f1[y, x] = 0
        # print(f1)
        f2 = f1
        for y in range(rows - 1, -1, -1):
            for x in range(cols - 1, -1, -1):
                if f1[y, x] == 0:
                    f2[y, x] = 0
                else:
                    f2[y, x] = min(f1[y, x], f2[y, x + 1] + 1)
        # print(f2**2)

        f3 = np.zeros(self.img.shape)
        for x in range(cols):
            for y in range(rows):
                min_v = float('inf')
                for j in range(rows):
                    value = f2[j, x]**2 + (y - j)**2
                    if value < min_v:
                        min_v = value

                f3[y, x] = min_v

        # print(f3)
        plt.imshow(f3, cmap="YlOrRd")
        plt.colorbar()
        # plt.show()
        plt.savefig(
            "output_images/distance_transform_{}".format(img_name))
        plt.close()
