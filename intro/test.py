import numpy as np
import cv2
from matplotlib import pyplot as plt


main_img = cv2.imread("me_original.jpg")
gray_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('main', main_img)
cv2.imshow('gray', gray_img)
plt.hist(gray_img.ravel(),256,[0,256])
plt.show()
hist,bins = np.histogram(gray_img.ravel(),256,[0,256])
rows, cols = gray_img.shape
abs_freq = hist
relative_freq = abs_freq / (rows*cols)
print(abs_freq)
print(bins)
print(relative_freq)
print(np.sum(relative_freq))


cv2.waitKey(0)
