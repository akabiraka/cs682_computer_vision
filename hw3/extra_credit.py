import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
import time
from skimage import data, filters


def scale(X, x_min, x_max):
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


def get_gradients(img2D):
    """
    Compute gradients in X and Y-axis
    img2D: input image as 2-dimension
    """
    img2D = cv2.GaussianBlur(img2D, (5, 5), 0)
    gx = cv2.Sobel(img2D, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img2D, cv2.CV_64F, 0, 1, ksize=3)
    return gx, gy


def get_bgrimg_gradients(bgrimg):
    """
    Compute edges of a color image.
    args:
        bgrimg: color image
    returns:
        B_gx: derivative in X-axis for B channel
        B_gy: derivative in Y-axis for B channel
        G_gx: derivative in X-axis for G channel
        G_gy: derivative in Y-axis for G channel
        R_gx: derivative in X-axis for R channel
        R_gy: derivative in Y-axis for R channel
    """
    bx, by = get_gradients(bgrimg[:, :, 0])
    gx, gy = get_gradients(bgrimg[:, :, 1])
    rx, ry = get_gradients(bgrimg[:, :, 2])
    return bx, by, gx, gy, rx, ry


def compute_structure_matrix(bx, by, gx, gy, rx, ry):
    """
    Compute the Structure(S) matrix components from Jacobian(J)
                            r_x     r_y
                        J = g_x     g_y
                            b_x     b_y

                        S = a b
                            b c

                        a = r_x^2 + g_x^2 + b_x^2
                        b = r_x*r_y + g_x*g_y + b_x*b_y
                        c = r_y^2 + g_y^2 + b_y^2
    returns:
        a,b,c element of S
    """
    a = np.square(rx) + np.square(gx) + np.square(bx)
    b = rx * ry + gx * gy + bx * by
    c = np.square(ry) + np.square(gy) + np.square(by)
    return a, b, c


def compute_eigen_values(a, b, c):
    """
    Compute eigen values from Structure matric component following slide.
    returns:
        lambda1: eigenvalue
        lambda2: eigenvalue, always 0
    """
    a_plus_c = a + c
    a_plus_c_sqr = np.square(a_plus_c)
    ac_sub_b_sqr = a * c - np.square(b)
    sqrt_a_plus_c_sqr_sub_4ac_sub_b_sqr = np.sqrt(
        a_plus_c_sqr - 4 * ac_sub_b_sqr)
    lambda1 = (a_plus_c + sqrt_a_plus_c_sqr_sub_4ac_sub_b_sqr) / 2
    lambda2 = (a_plus_c - sqrt_a_plus_c_sqr_sub_4ac_sub_b_sqr) / 2
    return lambda1, lambda2


def get_edge_direction(a, b, lambda1):
    """
    Returns edge direction by calculating eigenvectors.
    From slide, (a-lambda1) + by = 0
    => y/x = (a-lambda1) / -b
    let, y = a-lambda1
        so, x = -b
    returns:
        X-direction
        Y-direction
    """
    # x = - b / (a - lambda1)
    # x[np.isnan(x)] = 0
    # y = np.ones((1200, 1600))
    # y[np.isnan(x)] = 0
    #
    # denom = np.sqrt(np.square(x) + 1)
    # dx = x / denom
    # dy = 1.0 / denom
    # return dx, dy
    return -b, a - lambda1


def quiver_visualization(dx, dy, title):
    """
    Plot quiver visualization plot given dx, dy
    """
    step = 50
    fig, ax = plt.subplots(figsize=(8, 8))
    dx = dx[::step, ::step]
    dy = dy[::step, ::step]
    ax.quiver(dx, dy)
    ax.set_aspect('equal')
    ax.title.set_text(title)


def quiver_for_color(bx, by, gx, gy, rx, ry):
    """
    Calculate gradients of color image given gradients in b, g, r channel in X and Y-axis.
    """
    dx = bx + gx + rx
    dy = by + gy + ry
    quiver_visualization(
        dx, dy, title="Gradient visualization for color image")


def quiver_for_gray(img):
    """
    Calculate gradients of gray image given gradients in X and Y-axis.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx, dy = get_gradients(gray_img)
    quiver_visualization(
        dx, dy, title="Gradient visualization for grayscale image")


def non_maximum_suppression(magnitudes, angles):
    """
    Does non-maximum suppresion based on gradient magnitudes and orientation
    returns:
        gmax: the activation after non-maximum-suppression
    """
    gmax = np.zeros(magnitudes.shape)
    for i in range(gmax.shape[0]):
        for j in range(gmax.shape[1]):
            if angles[i][j] < 0:
                angles[i][j] += 360

            if ((j + 1) < gmax.shape[1]) and ((j - 1) >= 0) and ((i + 1) < gmax.shape[0]) and ((i - 1) >= 0):
                # 0 degrees
                if (angles[i][j] >= 337.5 or angles[i][j] < 22.5) or (angles[i][j] >= 157.5 and angles[i][j] < 202.5):
                    if magnitudes[i][j] >= magnitudes[i][j + 1] and magnitudes[i][j] >= magnitudes[i][j - 1]:
                        gmax[i][j] = magnitudes[i][j]
                # 45 degrees
                if (angles[i][j] >= 22.5 and angles[i][j] < 67.5) or (angles[i][j] >= 202.5 and angles[i][j] < 247.5):
                    if magnitudes[i][j] >= magnitudes[i - 1][j + 1] and magnitudes[i][j] >= magnitudes[i + 1][j - 1]:
                        gmax[i][j] = magnitudes[i][j]
                # 90 degrees
                if (angles[i][j] >= 67.5 and angles[i][j] < 112.5) or (angles[i][j] >= 247.5 and angles[i][j] < 292.5):
                    if magnitudes[i][j] >= magnitudes[i - 1][j] and magnitudes[i][j] >= magnitudes[i + 1][j]:
                        gmax[i][j] = magnitudes[i][j]
                # 135 degrees
                if (angles[i][j] >= 112.5 and angles[i][j] < 157.5) or (angles[i][j] >= 292.5 and angles[i][j] < 337.5):
                    if magnitudes[i][j] >= magnitudes[i - 1][j - 1] and magnitudes[i][j] >= magnitudes[i + 1][j + 1]:
                        gmax[i][j] = magnitudes[i][j]
    return gmax


def plot_hist(hist1, label1="label"):
    """
    Plots a histogram. Helper function for analysing.
    """
    plt.plot(hist1, label=label1)
    plt.show()


directory = 'ST2MainHall4/'
# filename = 'ST2MainHall4001.jpg'
filename = 'human3.tif'
img = cv2.imread(directory + filename)
cv2.imshow("Original image", img)
# Gradient visualization for grayscale image
quiver_for_gray(img)

# gradients for color image for every channel
bx, by, gx, gy, rx, ry = get_bgrimg_gradients(img)
# Gradient visualization for color image
quiver_for_color(bx, by, gx, gy, rx, ry)
# compute structure matrix
a, b, c = compute_structure_matrix(bx, by, gx, gy, rx, ry)
# trace, edge_strengths
magnitudes = a + c
# eigenvalues
lambda1, lambda2 = compute_eigen_values(
    a, b, c)
# eigen vectors orientation
dx, dy = get_edge_direction(a, b, lambda1)
# magnitude and angles of eigenvectors
magnitudes, angles = cv2.cartToPolar(dx, dy, angleInDegrees=True)
# scalling magnitudes from 0 to 255
magnitudes = scale(magnitudes, 0, 255)

# edges after non_maximum_suppression
edges = non_maximum_suppression(magnitudes, angles)
# cv2.imshow("after non max", edges)
# print(edges.min())
# print(edges.max())
# hist, bins = np.histogram(edges.ravel(), bins=256)
# hist[0:127] = 0
# print(hist)
# plot_hist(hist)

# appling hysteresis threshold
low = 0
high = 1.7
highest = (edges > high).astype(int)
our_edges = filters.apply_hysteresis_threshold(edges, low, high)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(highest + our_edges, cmap='magma')
plt.tight_layout()

plt.show()
cv2.waitKey(0)
