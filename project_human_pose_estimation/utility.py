import math
import matplotlib.pyplot as plt


def plot_images(images, img_name, titles=None, cols=3):
    rows = math.ceil(len(images) / cols)
    for i in range(len(images)):
        index = i + 1
        plt.subplot(rows, cols, index)
        plt.imshow(images[i])
        if titles is not None:
            plt.title(img_name + ": " + str(titles[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def plot_image_with_points(images, pts_list, cols=3):
    rows = math.ceil(len(images) / cols)
    # for i in range(len(images)):
    #     index = i + 1
    #     plt.subplot(rows, cols, index)
    #     plt.imshow(images[i])
    #     raw_pts = pts_list[i]
    #     plt.scatter(raw_pts[:, 0], raw_pts[:, 1], c='r', s=10)
    #     plt.xticks([])
    #     plt.yticks([])
    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
    k = 0
    for i in range(rows):
        for j in range(cols):
            raw_pts = pts_list[k]
            axs[i, j].imshow(images[k])
            axs[i, j].scatter(raw_pts[:, 0], raw_pts[:, 1], c='r', s=10)
            k += 1
    # plt.show()
    plt.savefig("output_images/raw_img_with_raw_pt.jpg")
