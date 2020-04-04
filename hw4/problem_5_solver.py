import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt


class Problem5Solver(object):
    def __init__(self):
        self.sample_img1 = np.zeros((8, 8))
        self.sample_img1[2, 0] = 1
        self.sample_img1[4, 4] = 1
        self.sample_img1[5, 4] = 1
        self.sample_img1[6, 4] = 1
        self.sample_img1[6, 5] = 1
        self.sample_img1[6, 6] = 1
        self.sample_img1[5, 7] = 1
        self.sample_img1[7, 7] = 1

        # forward, backward
        self.FB = np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]])

        self.adjacent_8 = [[-1, -1], [-1, 0], [-1, 1],
                           [0, -1], [0, 0], [0, 1],
                           [1, -1], [1, 0], [1, 1]]

    def __run_forward(self, img):
        rows, cols = img.shape
        for r in range(0, rows):
            for c in range(0, cols):
                values = []
                values.append(img[r, c])
                r_ = 1
                c_ = 1
                for q in range(4):
                    a = self.adjacent_8[q][0]
                    b = self.adjacent_8[q][1]
                    new_r = r + a
                    new_c = c + b
                    if(new_r >= 0 and new_c >= 0 and new_c < cols and new_r < rows):
                        value = img[new_r, new_c] * \
                            self.FB[r_ + a, c_ + b] + 1
                        if not math.isnan(value):
                            values.append(value)
                img[r, c] = np.array(values).min()

        return img

    def __run_backward(self, img):
        rows, cols = img.shape
        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):
                values = []
                values.append(img[r, c])
                r_ = 1
                c_ = 1
                for q in range(5, 9):
                    a = self.adjacent_8[q][0]
                    b = self.adjacent_8[q][1]
                    new_r = r + a
                    new_c = c + b
                    if(new_r >= 0 and new_c >= 0 and new_c < cols and new_r < rows):
                        value = img[new_r, new_c] * \
                            self.FB[r_ + a, c_ + b] + 1
                        if not math.isnan(value):
                            values.append(value)
                img[r, c] = np.array(values).min()
        return img

    def __compute_match_score(self, dist_matrix, template):
        template_mask = np.where(template > 0, True, False)  # template > 0
        return np.sum(dist_matrix[template_mask])

    def solve(self):
        # img1 = self.sample_img1
        # img2 = self.sample_img1
        # img = np.where(img1 > 0, 0, np.inf)
        # after_forward = self.__run_forward(img)
        # after_backward = self.__run_backward(after_forward)
        # match_score = self.__compute_match_score(after_backward, img2)
        # print(match_score)

        directory = 'GaitImages/'
        all_templates = []
        all_distance_transforms = []
        for filename in os.listdir(directory):
            print(filename)
            img = cv2.imread(directory + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.where(img > 0, 0, np.inf)

            after_forward = self.__run_forward(img)
            after_backward = self.__run_backward(after_forward)
            all_distance_transforms.append(after_backward)
            img = np.where(img == 0, 1, 0)
            all_templates.append(img)

        num_of_imgs = len(all_templates)
        all_match_scores = np.zeros(shape=(num_of_imgs, num_of_imgs))
        for i in range(num_of_imgs):
            for j in range(num_of_imgs):
                match_score = self.__compute_match_score(
                    all_distance_transforms[j], all_templates[i])
                print(
                    "i, j: {}, {} ------------ match_score: {}".format(i, j, match_score))
                all_match_scores[i, j] = match_score

        plt.imshow(all_match_scores)
        plt.colorbar()
        plt.title("Chamfer matching")
        # plt.show()
        plt.savefig("output_images/chamfer_matching.png")
        plt.close()
