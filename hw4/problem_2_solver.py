import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
from problem_1_solver import Problem1Solver


class Problem2Solver(object):
    """docstring for Problem3."""

    def __init__(self):
        super(Problem2Solver, self).__init__()
        self.p1 = Problem1Solver()

    def solve(self):
        for_n_images = 10
        directory = 'GaitImages/'
        all_contour_imgs = []
        all_poly_approximation_imgs = []
        all_convex_hull_imgs = []
        all_convexity_deficit_imgs = []
        all_convexity_deficit_areas = []
        i = 1
        for filename in os.listdir(directory):
            print(filename)
            img = cv2.imread(directory + filename)
            contours, img_with_contours = self.p1.compute_contours(img)
            all_contour_imgs.append(img_with_contours)
            poly_approximation, img_with_poly_approximation = self.p1.compute_polygonal_approximation(
                img, contours)
            all_poly_approximation_imgs.append(img_with_poly_approximation)
            hull_points, img_with_hull, img_with_contours_and_hull = self.p1.compute_convex_hull(
                img, contours, img_with_contours)
            all_convex_hull_imgs.append(img_with_hull)
            convexity_deficits, img_with_convexity_deficits, deficit_area = self.p1.compute_convexity_deficits(
                contours, img)
            all_convexity_deficit_imgs.append(img_with_convexity_deficits)
            all_convexity_deficit_areas.append(deficit_area)
            if i == for_n_images:
                break
            i += 1

        cols = math.floor(for_n_images / 2)
        self.__plot_images(
            all_contour_imgs, img_name="all_contours", cols=cols)
        self.__plot_images(all_poly_approximation_imgs,
                           img_name="all_poly_approximations", cols=cols)
        self.__plot_images(
            all_convex_hull_imgs, img_name="all_convex_hulls", cols=cols)
        self.__plot_images(all_convexity_deficit_imgs, img_name="all_convexity_deficits",
                           titles=all_convexity_deficit_areas, cols=cols)

    def __plot_images(self, images, img_name, titles=None, cols=3):

        rows = math.ceil(len(images) / cols)

        for i in range(len(images)):
            index = i + 1
            plt.subplot(rows, cols, index)
            plt.imshow(images[i], cmap='gray')
            if titles is not None:
                plt.title(str(titles[i]))
            plt.xticks([])
            plt.yticks([])
        # plt.show()
        plt.savefig("output_images/{}.png".format(img_name))
        plt.close()
