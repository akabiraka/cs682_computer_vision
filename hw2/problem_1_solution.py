from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_histogram(img):
    color = ('r', 'g', 'b')
    color_names = ['Red', 'Green', 'Blue']
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col, label=color_names[i])
        plt.xlim([-5, 260])
    plt.legend()
    plt.show()


# the following lines are necessary for draw_img_and_hist() function
plt.ion()
fig = plt.figure(figsize=(10, 3))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


def draw_img_and_hist(img):
    ax1.imshow(img)
    ax2.cla()
    color = ('r', 'g', 'b')
    color_names = ['Red', 'Green', 'Blue']
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax2.plot(histr, color=col, label=color_names[i])
        plt.xlim([-5, 260])
    plt.legend()
    plt.draw()
    plt.pause(0.001)


def on_click(event, x, y, flags, params):
    print("\nPress any key to stop.\n")
    if event == cv2.EVENT_MOUSEMOVE:  # EVENT_LBUTTONDOWN
        print('x = %d, y = %d' % (x, y))  # x = cols, y = rows
        B = int(I[y, x, 0])
        G = int(I[y, x, 1])
        R = int(I[y, x, 2])
        print('R=%d, G=%d, B=%d' % (R, G, B))
        intensity = (R + G + B) / 3
        print("Intensity: %f" % (intensity))
        rows, cols = I.shape[:2]
        w_rows, w_cols = 13, 13
        k_row, k_col = int(w_rows / 2), int(w_cols / 2)
        if(x - k_col < 0 or y - k_row < 0 or x + k_col > cols or y + k_row > rows):
            print("out of boundary\n")
            return
        else:
            window = I[y - k_row:y + k_row + 1, x - k_col:x + k_col + 1]
            window[:1, :] = 0  # 1st row
            window[12:, :] = 0  # last row
            window[:, :1] = 0  # 1st col
            window[:, 12:] = 0  # last col
            draw_img_and_hist(window)

        only_window = window[1:12, 1:12]
        # cv2.imshow("Window", only_window)
        mean, stddev = cv2.meanStdDev(only_window)
        print("Mean: ", mean)
        print("Standard Deviation: ", stddev)
        # draw_histogram(only_window)


def main():
    global I
    root = Tk()
    root.withdraw()  # to hide small tk window
    path = filedialog.askopenfilename()
    I = cv2.imread(path)
    cv2.imshow("Image", I)
    # draw_histogram(I)
    cv2.setMouseCallback('Image', on_click)


main()
plt.show()
cv2.waitKey(0)
