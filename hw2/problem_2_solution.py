import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
import time

start = time.time()


def draw_color_histogram():
    # 'ST2MainHall4/ST2MainHall4001.jpg'
    directory = 'ST2MainHall4/'
    all_hists = []
    label = []
    for filename in os.listdir(directory):
        print(filename)
        temp_img = cv2.imread(directory + filename)
        # temp_img = cv2.imread('ST2MainHall4/ST2MainHall4001.jpg')
        img = np.zeros(shape=temp_img.shape, dtype=np.uint16)
        img = img + temp_img
        indexes = ((img[:, :, 0] >> 5) << 6) + \
            ((img[:, :, 1] >> 5) << 3) + (img[:, :, 2] >> 5)
        # hist,bins = np.histogram(indexes.ravel(),512,[0,512])
        all_hists.append(indexes.ravel())
        label.append(filename)

    bins = np.linspace(-5, 520, 15)
    # print(bins)
    plt.hist(all_hists, bins=bins)
    # plt.legend(loc="upper right")
    plt.title('Histograms')
    plt.show()


def scale(X, x_min, x_max):
    mat_min = X.min()
    mat_max = X.max()
    nom = (X - mat_min) * (x_max - x_min)
    denom = mat_max - mat_min
    return x_min + nom / denom


def compute_color_histogram(temp_img):
    img = np.zeros(shape=temp_img.shape, dtype=np.uint16)
    img = img + temp_img
    indexes = ((img[:, :, 0] >> 5) << 6) + \
        ((img[:, :, 1] >> 5) << 3) + (img[:, :, 2] >> 5)
    hist, bins = np.histogram(indexes.ravel(), 512, [0, 512])
    return hist, bins


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


directory = 'ST2MainHall4/'
all_hists = []
for filename in os.listdir(directory):
    print(filename)
    temp_img = cv2.imread(directory + filename)
    hist, _ = compute_color_histogram(temp_img)
    # print(hist.shape)
    all_hists.append(hist)

num_of_imgs = len(all_hists)
# confusion_matrix_using_hist_intersection
cmuhi = np.zeros(shape=(num_of_imgs, num_of_imgs))
# confusion_matrix_using_chi_squared_measure
cmucsm = np.zeros(shape=(num_of_imgs, num_of_imgs))
for i in range(num_of_imgs):
    for j in range(num_of_imgs):
        cmuhi[i, j] = hist_intersection(all_hists[i], all_hists[j])
        cmucsm[i, j] = chi_squared_measure(all_hists[i], all_hists[j])

cmuhi = scale(cmuhi, 0, 255)
cmucsm = scale(cmucsm, 0, 255)

fig, axes = plt.subplots(ncols=2, figsize=(20, 20))
ax1, ax2 = axes

im1 = ax1.matshow(cmuhi, cmap='coolwarm')
im2 = ax2.matshow(cmucsm, cmap='RdBu')

ax1.set_title('Similarity matrix using histogram intersection', y=-.1)
ax2.set_title('Similarity matrix using chi-square measure', y=-.1)

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

end = time.time()
print("Time: ", end - start)
plt.show()
cv2.waitKey(0)
