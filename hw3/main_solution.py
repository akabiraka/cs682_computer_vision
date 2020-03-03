import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
import time

start = time.time()


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


def plot_two_images(img1, img2, title1='image1', title2='Image2'):
    """
    Plots two images sided by side.
    img1: left image of the plot
    img2: right image of the plot.
    """
    plt.subplot(121), plt.imshow(img1, cmap='gray')
    plt.title('Image1'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.title('Image2'), plt.xticks([]), plt.yticks([])
    plt.show()


def get_gradients(img):
    """
    Compute gradients for each pixel of an image
    args:
        img: given image of single channel
    returns:
        gx: derivative in X-axis (horizontal)
        gy: derivative in Y-axis (vertical)
        g: magnitude
        theta: angle or orientation
    """
    # smoothing is already in sobel operator
    img = cv2.GaussianBlur(img, (5, 5), 0)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    plot_two_images(gx, gy)
    # edge_gradient = np.absolute(gx) + np.absolute(gy)  # using L-1 norm
    # g = np.sqrt(np.square(gx) + np.square(gy))  # using L-2 norm
    # theta = np.arctan(gx / gy)
    g, theta = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # print(theta)
    # plot_two_images(theta, g)
    return gx, gy, g, theta


def compute_gray_label_edges(rgbimg):
    """
    Compute edges of a gray image.
    args:
        img: given grayscale image
    returns:
        gx: derivative in X-axis (horizontal)
        gy: derivative in Y-axis (vertical)
        g: magnitude
        theta: angle or orientation
        gray_edges: edges computed by canny operator
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_edges = cv2.Canny(gray_img, 50, 180, L2gradient=False)
    gx, gy, g, theta = get_gradients(gray_img)
    # plot_two_images(gray_img, gray_edges)
    return gx, gy, g, theta, gray_edges


def get_channel_wise_edges(color_img, channel_index):
    """
    Compute edges of a color image.
    args:
        color_img: given color image
        channel_index: index of the channel
    returns:
        gx: derivative in X-axis (horizontal)
        gy: derivative in Y-axis (vertical)
        g: magnitude
        theta: angle or orientation
        gray_edges: edges computed by canny operator
    """
    img = color_img[:, :, channel_index]
    edges = cv2.Canny(img, 40, 150, L2gradient=True)
    gx, gy, g, theta = get_gradients(img)
    return gx, gy, g, theta, edges


def compute_color_label_edges(rgbimg):
    """
    Compute edges of a color image.
    args:
        rgbimg: color image
    returns:
        B_gx: derivative in X-axis for B channel
        B_gy: derivative in Y-axis for B channel
        G_gx: derivative in X-axis for G channel
        G_gy: derivative in Y-axis for G channel
        R_gx: derivative in X-axis for R channel
        R_gy: derivative in Y-axis for R channel
    """
    B_gx, B_gy, B_g, B_theta, B_edges = get_channel_wise_edges(
        rgbimg, channel_index=0)
    G_gx, G_gy, G_g, G_theta, G_edges = get_channel_wise_edges(
        rgbimg, channel_index=1)
    R_gx, R_gy, R_g, R_theta, R_edges = get_channel_wise_edges(
        rgbimg, channel_index=2)

    plot_two_images(rgbimg[:, :, 0], B_edges)
    plot_two_images(rgbimg[:, :, 1], G_edges)
    plot_two_images(rgbimg[:, :, 2], R_edges)
    return B_gx, B_gy, G_gx, G_gy, R_gx, R_gy


def get_edge_hist(magnitudes, angles, scaling=True, scale_min=1, scale_max=100):
    """
    Compute edge histograms by considering magnitudes and not considering magnitudes and returns both
    """
    angles = np.floor(angles / 10).astype(np.int32)
    # count all selected edges using the same value, say 1
    hist_without_contrib, bins = np.histogram(angles.ravel(), bins=36)
    rows, cols = angles.shape
    hist_with_contrib = np.zeros(36)
    # use the edge magnitude in the histograms
    for i in range(rows):
        for j in range(cols):
            hist_with_contrib[angles[i, j]
                              ] = hist_with_contrib[angles[i, j]] + magnitudes[i, j]
    if scaling == True:
        return scale(hist_without_contrib, scale_min, scale_max), scale(hist_with_contrib, scale_min, scale_max)
    return hist_without_contrib, hist_with_contrib


def plot_two_arrays(hist1, hist2, label1, label2, title):
    """
    Plot two histograms in the same plot.
    """
    plt.plot(hist1, label=label1)
    plt.plot(hist2, label=label2)
    plt.legend()
    plt.title(title)
    plt.show()


def gray_edge_hist(magnitudes, angles):
    """
    Compute histograms from gray scale images.
    """
    hist_without_contrib, hist_with_contrib = get_edge_hist(
        magnitudes, angles, scaling=True)
    plot_two_arrays(hist1=hist_without_contrib, hist2=hist_with_contrib, label1="Histograms without considering magnitudes",
                    label2="Histograms by considering magnitudes", title="Gray edge histogram")
    return hist_without_contrib, hist_with_contrib


def color_edge_hist(B_gx, B_gy, G_gx, G_gy, R_gx, R_gy):
    """
    Compute histograms from color scale images.
    """
    gx = B_gx + G_gx + R_gx
    gy = B_gy + G_gy + R_gy
    angles = ((np.arctan2(gy, gx) * 180 / np.pi) + 360) % 360
    magnitudes = np.sqrt(np.square(B_gx) + np.square(G_gx) + np.square(R_gx) +
                         np.square(B_gy) + np.square(G_gy) + np.square(R_gy))
    hist_without_contrib, hist_with_contrib = get_edge_hist(
        magnitudes, angles, scaling=True)
    plot_two_arrays(hist1=hist_without_contrib, hist2=hist_with_contrib, label1="Histograms without considering magnitudes",
                    label2="Histograms by considering magnitudes", title="Color edge histogram")
    return hist_without_contrib, hist_with_contrib


def hist_intersection(hist1, hist2):
    sum_of_min, sum_of_max = 0, 0
    for i in range(hist1.shape[0]):
        sum_of_min += min(hist1[i], hist2[i])
        sum_of_max += max(hist1[i], hist2[i])
    return sum_of_min / sum_of_max


def chi_squared_measure(hist1, hist2):
    chi_square = 0
    for i in range(hist1.shape[0]):
        if hist1[i] + hist2[i] > 0:
            chi_square += ((hist1[i] - hist2[i])**2) / (hist1[i] + hist2[i])
    return chi_square


def plot_hist_comparison(all_hists):
    num_of_imgs = len(all_hists)
    # confusion matrix using hist intersection
    cm_hi = np.zeros(shape=(num_of_imgs, num_of_imgs))
    # confusion matrix using chi squared measure
    cm_csm = np.zeros(shape=(num_of_imgs, num_of_imgs))

    num_of_imgs = len(all_hists)
    for i in range(num_of_imgs):
        for j in range(num_of_imgs):
            cm_hi[i, j] = hist_intersection(all_hists[i], all_hists[j])
            cm_csm[i, j] = chi_squared_measure(all_hists[i], all_hists[j])

    cm_hi = scale(cm_hi, 0, 255)
    cm_csm = scale(cm_csm, 0, 255)

    fig, axes = plt.subplots(ncols=2, figsize=(20, 20))
    ax1, ax2 = axes

    im1 = ax1.matshow(cm_hi, cmap='coolwarm')
    im2 = ax2.matshow(cm_csm, cmap='RdBu')

    ax1.set_title('Similarity matrix using histogram intersection', y=-.1)
    ax2.set_title('Similarity matrix using chi-square measure', y=-.1)

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    plt.show()

# directory = 'ST2MainHall4/'
# filename = 'ST2MainHall4001.jpg'
# img = cv2.imread(directory + filename)
# cv2.imshow("image", img)
# gx, gy, g, theta, gray_edges = compute_gray_label_edges(img)
# gray_edge_hist(g, theta)
#
# B_gx, B_gy, G_gx, G_gy, R_gx, R_gy = compute_color_label_edges(img)
# color_edge_hist(B_gx, B_gy, G_gx, G_gy, R_gx, R_gy)


# directory = 'ST2MainHall4/'
directory = 'sample_images/'
all_gray_edge_hists = []
all_color_edge_hists = []
for filename in os.listdir(directory):
    print(filename)
    img = cv2.imread(directory + filename)
    gx, gy, g, theta, gray_edges = compute_gray_label_edges(img)
    hist_without_contrib, hist_with_contrib = gray_edge_hist(g, theta)
    all_gray_edge_hists.append(hist_with_contrib)

    B_gx, B_gy, G_gx, G_gy, R_gx, R_gy = compute_color_label_edges(img)
    hist_without_contrib, hist_with_contrib = color_edge_hist(
        B_gx, B_gy, G_gx, G_gy, R_gx, R_gy)
    all_color_edge_hists.append(hist_with_contrib)


plot_hist_comparison(all_gray_edge_hists)
plot_hist_comparison(all_color_edge_hists)

end = time.time()
print("Time: ", end - start)
cv2.waitKey(0)
