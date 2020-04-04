import numpy as np
from numpy.polynomial import Polynomial as P
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import math
import warnings


class Problem3Solver(object):
    """docstring for Problem3."""

    def __init__(self, boundaries=None, binary_img=None):
        super(Problem3Solver, self).__init__()
        self.boundaries = boundaries
        self.binary_img = binary_img

    def scale(self, X, x_min, x_max):
        """
        Scale a vector or matrix between given min and max value.
        X: array or matrix
        x_min: minimum of X
        x_max: maximum of X
        """
        min = X.min()
        max = X.max()
        nom = (X - min) * (x_max - x_min)
        denom = max - min
        return x_min + nom / denom

    def __get_curvatures(self, x, y, w_size):
        curvatures = np.zeros(len(x))
        half_width = math.floor(w_size / 2)
        for k in range(half_width, len(x) - half_width):
            X = x[k - half_width: k + half_width]
            Y = y[k - half_width: k + half_width]
            t = range(k - half_width, k + half_width)
            coeff_a = np.polyfit(X, t, 2)
            coeff_b = np.polyfit(Y, t, 2)
            curvatures[k] = (2 * (coeff_a[1] * coeff_b[2] - coeff_a[2]
                                  * coeff_b[1])) / ((coeff_a[1]**2 + coeff_b[1]**2)**1.5)
        return curvatures

    def solve(self):
        boundaries = self.boundaries
        binary_img = self.binary_img

        x = boundaries[:, 0, 0]
        y = boundaries[:, 0, 1]
        # x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  #
        # y = np.array([1, 5, 3, 4, 5, 6, 7, 8, 9])  #

        w_sizes = range(3, 13, 2)
        for w_size in w_sizes:
            curvatures = self.__get_curvatures(x, y, w_size)
            curvatures = self.scale(curvatures, 0, 99)

            color_map = cm = plt.get_cmap('YlOrRd')
            cNorm = colors.Normalize(vmin=0, vmax=99)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)

            fig, axs = plt.subplots(2, 2, figsize=(20, 10))
            ax1 = axs[0, 0]
            ax1.plot(x, y, 'g', linewidth=2)
            ax1.set_title("Boundaries")

            ax2 = axs[0, 1]
            ax2.plot(x, curvatures, 'b-', y, curvatures, 'r--')
            ax2.set_title("Curvatures, window(k): {}".format(w_size))

            ax3 = axs[1, 0]
            for k in range(0, len(x)):
                colorVal = scalarMap.to_rgba(curvatures[k])
                ax3.plot(x[k], y[k], '.', color=colorVal,
                         linewidth=2, markersize=20)
            ax3.set_title("Curvatures hot map")

            if binary_img is not None:
                ax4 = axs[1, 1]
                for k in range(0, len(x)):
                    colorVal = scalarMap.to_rgba(curvatures[k])
                    ax4.plot(x[k], y[k], '.', color=colorVal,
                             linewidth=2, markersize=20)
                ax4.imshow(binary_img, cmap='gray', origin='lower')
                ax4.set_title("Image with curvature hot map")

            # plt.show()
            fig.savefig("output_images/using_window_{}.png".format(w_size))
            plt.close()


# def test_problem_3_solver(img):
#     # find and display boundaries/contours
#     binary_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     boundaries, hierarchy = cv2.findContours(
#         binary_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
#     boundaries = np.array(boundaries[0])
#     print("boundaries shape: ", np.array(boundaries).shape)  # 152x1x2
#     solve_problem_3(boundaries, binary_img)
#
#
# def solve_problem_3(boundaries, binary_img):
#     x = boundaries[:, 0, 0]
#     y = boundaries[:, 0, 1]
#     # x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  #
#     # y = np.array([1, 5, 3, 4, 5, 6, 7, 8, 9])  #
#
#     w_sizes = range(3, 13, 2)
#     for w_size in w_sizes:
#         print("==" * 80)
#         print("window k: ", w_size)
#         curvatures = np.zeros(len(x))
#         half_width = math.floor(w_size / 2)
#         for k in range(half_width, len(x) - half_width):
#             X = x[k - half_width: k + half_width]
#             Y = y[k - half_width: k + half_width]
#             t = range(k - half_width, k + half_width)
#             coeff_a = np.polyfit(X, t, 2)
#             coeff_b = np.polyfit(Y, t, 2)
#             curvatures[k] = (2 * (coeff_a[1] * coeff_b[2] - coeff_a[2]
#                                   * coeff_b[1])) / ((coeff_a[1]**2 + coeff_b[1]**2)**1.5)
#         print("==" * 80)
#
#         curvatures = scale(curvatures, 0, 99)
#
#         color_map = cm = plt.get_cmap('YlOrRd')
#         cNorm = colors.Normalize(vmin=0, vmax=99)
#         scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)
#
#         fig, axs = plt.subplots(2, 2, figsize=(20, 10))
#         ax1 = axs[0, 0]
#         ax1.plot(x, y, 'g', linewidth=2)
#         ax1.set_title("Boundaries")
#
#         ax2 = axs[0, 1]
#         ax2.plot(x, curvatures, 'b-', y, curvatures, 'r--')
#         ax2.set_title("Curvatures, window(k): {}".format(w_size))
#
#         ax3 = axs[1, 0]
#         for k in range(0, len(x)):
#             colorVal = scalarMap.to_rgba(curvatures[k])
#             ax3.plot(x[k], y[k], '.', color=colorVal,
#                      linewidth=2, markersize=20)
#         ax3.set_title("Curvatures hot map")
#
#         ax4 = axs[1, 1]
#         for k in range(0, len(x)):
#             colorVal = scalarMap.to_rgba(curvatures[k])
#             ax4.plot(x[k], y[k], '.', color=colorVal,
#                      linewidth=2, markersize=20)
#         ax4.imshow(binary_img, cmap='gray', origin='lower')
#         ax4.set_title("Image with curvature hot map")
#
#         # plt.show()
#         fig.savefig("output_images/using_window_{}.png".format(w_size))
#         plt.close()
#
#
# def get_curvatures(x, y):
#     w_sizes = range(5, 19, 2)
#     degrees = range(1, 10)
#
#     best_error = float("inf")
#     best_curvatures = np.zeros(len(x))
#     best_deg = 0
#     best_w_size = 0
#
#     for w_size in w_sizes:
#         half_width = math.floor(w_size / 2)
#         curvatures = np.zeros(len(x))
#
#         for deg in degrees:
#             total_error = 0.0
#
#             for k in range(half_width, len(x) - half_width):
#                 X = x[k - half_width: k + half_width]
#                 Y = y[k - half_width: k + half_width]
#                 coefficients = np.polyfit(X, Y, deg)
#                 error = np.sum((np.polyval(coefficients, X) - Y)**2)
#                 total_error += error
#                 curvatures[k] = coefficients[0]
#
#             if(total_error < best_error):
#                 best_error = total_error
#                 best_curvatures = curvatures
#                 best_deg = deg
#                 best_w_size = w_size
#
#     #         plt.subplot(3, 3, deg)
#     #         plt.plot(x, curvatures, 'b-', y, curvatures, 'r--')
#     #         plt.title('window: {}, deg: {}'.format(w_size, deg))
#     #     plt.show()
#     #
#     # plt.subplot(2, 1, 1)
#     # plt.plot(x, best_curvatures, 'b-', y, best_curvatures, 'r--')
#     # plt.title('Curvatures[{}, {}], window: {}, deg: {}'.format(
#     #     best_curvatures.min(), best_curvatures.max(), best_w_size, best_deg,))
#     #
#     # scaled_best_curvatures = scale(best_curvatures, 0, 90)
#     # plt.subplot(2, 1, 2)
#     # plt.plot(x, scaled_best_curvatures, 'b-', y, scaled_best_curvatures, 'r--')
#     # plt.title('Scaled curvatures [{}, {}], window: {}, deg: {}'.format(
#     #     scaled_best_curvatures.min(), scaled_best_curvatures.max(), best_w_size, best_deg))
#     #
#     # plt.show()
#
#     return best_error, best_curvatures, best_deg, best_w_size
#
#
# def solve_problem_3_again(boundaries, binary_img):
#     x = boundaries[:, 0, 0]
#     y = boundaries[:, 0, 1]
#
#     best_error, best_curvatures, best_deg, best_w_size = get_curvatures(
#         x, y)
#     best_curvatures = scale(best_curvatures, 0, 99)
#
#     color_map = cm = plt.get_cmap('YlOrRd')
#     cNorm = colors.Normalize(vmin=0, vmax=99)
#     scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)
#
#     plt.subplot(2, 2, 1)
#     plt.plot(x, y, 'g', linewidth=2)
#
#     plt.subplot(2, 2, 2)
#     plt.plot(x, best_curvatures, 'b-', y, best_curvatures, 'r--')
#
#     for k in range(0, len(x)):
#         colorVal = scalarMap.to_rgba(best_curvatures[k])
#         plt.subplot(2, 2, 3)
#         plt.plot(x[k], y[k], '.', color=colorVal, linewidth=2, markersize=20)
#
#     for k in range(0, len(x)):
#         colorVal = scalarMap.to_rgba(best_curvatures[k])
#         plt.subplot(2, 2, 4)
#         plt.plot(x[k], y[k], '.', color=colorVal, linewidth=2, markersize=20)
#
#     plt.imshow(binary_img, cmap='gray', origin='lower')
#     plt.show()
