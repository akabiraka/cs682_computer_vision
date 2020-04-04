import numpy as np
from numpy.polynomial import Polynomial as P
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import math
import warnings


class Problem1Solver(object):
    """docstring for Problem3."""

    def __init__(self):
        super(Problem1Solver, self).__init__()
        self.thickness = 2

    def solve(self, img):
        binary_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, img_with_contours = self.compute_contours(img)
        poly_approximation, img_with_poly_approximation = self.compute_polygonal_approximation(
            img, contours)
        hull_points, img_with_hull, img_with_contours_and_hull = self.compute_convex_hull(
            img, contours, img_with_contours)
        convexity_deficits, img_with_convexity_deficits, convexity_deficit_area = self.compute_convexity_deficits(
            contours, img)
        convexity_deficits, img_with_contours_hulls_and_convexity_deficits, convexity_deficit_area = self.compute_convexity_deficits(
            contours, img_with_contours_and_hull)
        original_area, original_perimeter = self.compute_area_perimeter(
            contours[0])
        print("original_area={:.1f}, original_perimeter={:.1f}".format(
            original_area, original_perimeter))
        hull_area, hull_perimeter = self.compute_area_perimeter(hull_points)
        print("hull_area={:.1f}, hull_perimeter={:.1f}".format(
            hull_area, hull_perimeter))
        m00, m10, m01, m20, m11, m02 = self.compute_moments(contours[0])
        print("Moments for Original image:\n m00={:.1f}, m10={:.1f}, m01={:.1f}, m20={:.1f}, m11={:.1f}, m02={:.1f}".format(
            m00, m10, m01, m20, m11, m02))
        m00, m10, m01, m20, m11, m02 = self.compute_moments(hull_points)
        print("Moments for Convex hulls:\n m00={:.1f}, m10={:.1f}, m01={:.1f}, m20={:.1f}, m11={:.1f}, m02={:.1f}".format(
            m00, m10, m01, m20, m11, m02))

        images = [binary_img, img_with_contours, img_with_poly_approximation,
                  img_with_contours_and_hull, img_with_convexity_deficits,
                  img_with_contours_hulls_and_convexity_deficits]
        titles = ["binary_img", "img_with_contours", "img_with_poly_approximation",
                  "img_with_contours_and_hull", "img with Convexity Deficit area: {:.1f}".format(
                      convexity_deficit_area),
                  "img_with contours, hulls, convexity_deficits area: {:.1f}".format(convexity_deficit_area)]

        self.__plot_images(images, titles)

    def compute_area_perimeter(self, contour_points):
        """
        contour_points's shape should be like [147x1x2]
        """
        area = cv2.contourArea(contour_points)
        perimeter = cv2.arcLength(contour_points, closed=True)
        return area, perimeter

    def compute_moments(self, contour_points):
        momemts = cv2.moments(contour_points, binaryImage=True)
        return momemts['m00'], momemts['m10'], momemts['m01'], momemts['m20'], momemts['m11'], momemts['m02']

    def compute_contours(self, img):
        # find and display boundaries/contours
        binary_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(
            binary_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        img_with_contours = cv2.drawContours(
            img.copy(), contours=contours, contourIdx=-1, color=self.__get_random_color(), thickness=self.thickness)
        # self.plot_two_images(img, img_with_contours, title1='Given image',
        #                 title2='Image with contours')
        # print("contours shape: ", np.array(contours).shape)  # (1, 152, 1, 2)
        return contours, img_with_contours

    def compute_polygonal_approximation(self, img, contours):
        # polygon approximation of computed boundaries
        epsilon = .01
        approxi_poly = cv2.approxPolyDP(
            curve=contours[0], epsilon=epsilon, closed=True)
        img_with_approxi_poly = cv2.drawContours(
            img.copy(), contours=[approxi_poly], contourIdx=-1, color=self.__get_random_color(), thickness=self.thickness)
        # self.plot_two_images(img, img_with_approxi_poly, title1='Given image',
        #                 title2='Image with polygon approximation')
        # print("approximate poly shape: ", np.array(approxi_poly).shape)  # (152, 1, 2)
        # self.plot_two_images(img_with_contours, img_with_approxi_poly, title1='Given image',
        #                 title2='Image with polygon approximation')
        return approxi_poly, img_with_approxi_poly

    def compute_convex_hull(self, img, contours, img_with_contours):
        # convex hull
        # returns convex hull points
        hull_points = cv2.convexHull(contours[0], returnPoints=True)
        img_with_hull = cv2.drawContours(
            img.copy(), contours=[hull_points], contourIdx=-1, color=self.__get_random_color(), thickness=self.thickness)
        # print("convex hull shape returnPoints=True: ",
        #       np.array(hull_points).shape)  # (19, 1, 2)
        # self.plot_two_images(img, img_with_hull, title1='Given image',
        #                 title2='Image with convex hull')
        img_with_contours_and_hull = cv2.drawContours(
            img_with_contours.copy(), contours=[hull_points], contourIdx=-1, color=self.__get_random_color(), thickness=self.thickness)
        # self.plot_two_images(img, img_with_contours_and_hull, title1='Given image',
        #                 title2='Image with contours and convex hull')
        return hull_points, img_with_hull, img_with_contours_and_hull

    def compute_area(self, pt_start, pt_end, pt_far):
        x1 = pt_start[0]
        y1 = pt_start[1]
        x2 = pt_end[0]
        y2 = pt_end[1]
        x3 = pt_far[0]
        y3 = pt_far[1]
        area = abs(0.5 * (((x2 - x1) * (y3 - y1)) - ((x3 - x1) * (y2 - y1))))
        return area

    def compute_convexity_deficits(self, contours, img):
        hull_indices = cv2.convexHull(contours[0], returnPoints=False)
        # print("convex hull shape returnPoints=False: ",
        #       np.array(hull_indices).shape)
        convexity_deficits = cv2.convexityDefects(contours[0], hull_indices)

        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        canvas = FigureCanvasAgg(fig)
        plt.imshow(img)
        contours = contours[0]
        convexity_deficit_area = 0.0
        for c in range(len(convexity_deficits)):
            start_index = convexity_deficits[c][0][0]
            end_index = convexity_deficits[c][0][1]
            far_pt_index = convexity_deficits[c][0][2]
            depth = convexity_deficits[c][0][3] / 256

            pt_start = contours[start_index][0]
            pt_end = contours[end_index][0]
            pt_far = contours[far_pt_index][0]

            x_values, y_values = self.get_xy_values(pt_start, pt_end)
            plt.plot(x_values, y_values, 'r-', linewidth=3.0, markersize=12)
            x_values, y_values = self.get_xy_values(pt_start, pt_far)
            plt.plot(x_values, y_values, 'r-', linewidth=3.0, markersize=12)
            x_values, y_values = self.get_xy_values(pt_end, pt_far)
            plt.plot(x_values, y_values, 'r-', linewidth=3.0, markersize=12)
            circle = plt.Circle(pt_far, color='g', fill=False)
            ax.add_artist(circle)
            convexity_deficit_area += self.compute_area(
                pt_start, pt_end, pt_far)

        ax.set(frame_on=False)
        fig.tight_layout()
        plt.close()
        canvas.draw()
        img = canvas.buffer_rgba()
        return convexity_deficits, img, convexity_deficit_area

    def get_xy_values(self, pt1, pt2):
        x_values = [pt1[0], pt2[0]]
        y_values = [pt1[1], pt2[1]]
        return x_values, y_values

    def solve_(self, img):
        # find and display boundaries/contours
        binary_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(
            binary_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        img_with_contours = cv2.drawContours(
            img.copy(), contours=contours, contourIdx=-1, color=self.__get_random_color(), thickness=self.thickness)
        # self.plot_two_images(img, img_with_contours, title1='Given image',
        #                 title2='Image with contours')
        print("contours shape: ", np.array(contours).shape)  # (1, 152, 1, 2)

        # polygon approximation of computed boundaries
        epsilon = .01
        approxi_poly = cv2.approxPolyDP(
            curve=contours[0], epsilon=epsilon, closed=True)
        img_with_approxi_poly = cv2.drawContours(
            img.copy(), contours=[approxi_poly], contourIdx=-1, color=self.__get_random_color(), thickness=self.thickness)
        # self.plot_two_images(img, img_with_approxi_poly, title1='Given image',
        #                 title2='Image with polygon approximation')
        print("approximate poly shape: ", np.array(
            approxi_poly).shape)  # (152, 1, 2)
        # self.plot_two_images(img_with_contours, img_with_approxi_poly, title1='Given image',
        #                 title2='Image with polygon approximation')

        # convex hull
        # returns convex hull points
        hull_points = cv2.convexHull(contours[0], returnPoints=True)
        img_with_hull = cv2.drawContours(
            img.copy(), contours=[hull_points], contourIdx=-1, color=self.__get_random_color(), thickness=self.thickness)
        print("convex hull shape returnPoints=True: ",
              np.array(hull_points).shape)  # (19, 1, 2)
        # self.plot_two_images(img, img_with_hull, title1='Given image',
        #                 title2='Image with convex hull')
        img_with_contours_and_hull = cv2.drawContours(
            img_with_contours.copy(), contours=[hull_points], contourIdx=-1, color=self.__get_random_color(), thickness=self.thickness)
        # self.plot_two_images(img, img_with_contours_and_hull, title1='Given image',
        #                 title2='Image with contours and convex hull')

        # deficits of convexity
        # returns indices of the convex hull points.
        hull_indices = cv2.convexHull(contours[0], returnPoints=False)
        print("convex hull shape returnPoints=False: ",
              np.array(hull_indices).shape)
        convexity_deficits = cv2.convexityDefects(contours[0], hull_indices)
        # (12, 1, 4): (start_index, end_index, farthest_pt_index, fixpt_depth),
        # where indices are 0-based indices in the original contour
        print("convexity deficits shape: ", np.array(convexity_deficits).shape)

        # area
        original_area = cv2.contourArea(contours[0])
        print("original cortour area: ", original_area)
        hull_area = cv2.contourArea(hull_points)
        print("hull area: ", hull_area)

        # perimeter
        original_perimeter = cv2.arcLength(contours[0], closed=True)
        print("original perimeter: ", original_perimeter)
        hull_perimeter = cv2.arcLength(hull_points, closed=True)
        print("hull perimeter: ", hull_perimeter)

        # moments
        original_momemts = cv2.moments(binary_img, binaryImage=True)
        print("original_momemts: ", original_momemts)
        hull_moments = cv2.moments(hull_points, binaryImage=True)
        print("hull_moments: ", hull_moments)

        img_with_approxi_poly_and_hull = cv2.drawContours(
            img_with_approxi_poly.copy(), contours=[hull_points], contourIdx=-1, color=self.__get_random_color(), thickness=self.thickness)
        # self.plot_two_images(img, img_with_contours_and_hull, title1='Given image',
        #                 title2='Image with polygonal approximation and convex hull')

        images = [binary_img, img_with_contours, img_with_approxi_poly,
                  img_with_contours_and_hull, img_with_approxi_poly_and_hull]
        titles = ["binary_img", "img_with_contours", "img_with_approxi_poly",
                  "img_with_contours_and_hull", "img_with_approxi_poly_and_hull"]
        self.__plot_images(images, titles)

        # solution 2 start
        total_deficit_area = hull_area - original_area
        print("total deficit area: ", total_deficit_area)
        print("convexity deficits points: ", convexity_deficits)

    def __get_random_color(self):
        color = list(np.random.random(size=3) * 256)
        return color

    def __plot_images(self, images, titles, cols=3):
        if(len(images) != len(titles)):
            print("image and titles do no match!!!")
            return

        rows = math.ceil(len(images) / cols)

        for i in range(len(images)):
            index = i + 1
            plt.subplot(rows, cols, index)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i]), plt.xticks([]), plt.yticks([])
        plt.show()

    def __plot_two_images(self, img1, img2, title1='image1', title2='Image2'):
        """
        Plots two images sided by side.
        img1: left image of the plot
        img2: right image of the plot.
        """
        plt.subplot(121), plt.imshow(img1, cmap='gray')
        plt.title(title1), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img2, cmap='gray')
        plt.title(title2), plt.xticks([]), plt.yticks([])
        plt.show()
