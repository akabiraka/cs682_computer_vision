
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_images(original_img, transformed_img, title):
    final_frame = cv2.hconcat((original_img, transformed_img))
    cv2.imshow(title, final_frame)
    cv2.waitKey(0)
    # plt.subplot(121),plt.imshow(original_img),plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(transformed_img),plt.title('Transformed')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

def blurring_img(img):
    kernel = (5, 5)
    # blurred_img = cv2.blur(img, kernel) # average blurring
    blurred_img = cv2.GaussianBlur(img, kernel, 0) # Gaussian blurring
    # blurred_img = cv2.medianBlur(img,5) # median blurring
    show_images(img, blurred_img, 'Blurred')
    # cv2.imshow("Transformed", blurred_img)
    # plt.show()

def rotating_img(img, isRGB):
    if isRGB:
        rows, cols = img.shape[:2]
    else:
        rows, cols = img.shape
    temp = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    rotated_img = cv2.warpAffine(img,temp,(cols,rows))
    show_images(img, rotated_img, 'Rotated')

def just_ops(color_img, gray_img):
    bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    plus = color_img + bgr_img
    minus = color_img - bgr_img + 100
    # minus_absolute = np.absolute(minus) + 100
    # cv2.imshow("aaa", minus)

def transform_affine(img, isRGB):
    if isRGB:
        rows,cols,ch = img.shape
    else:
        rows,cols = img.shape
    pt1 = np.float32([[50,50],[200,50],[50,200]])
    pt2 = np.float32([[10,100],[200,50],[100,250]])
    temp = cv2.getAffineTransform(pt1,pt2)
    affined = cv2.warpAffine(img,temp,(cols,rows))
    show_images(img, affined, 'affined')

def rgb2hsb(img):
    hsb_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    show_images(img, hsb_img, 'hsb')

def gray2rgb(img):
    bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imshow('gray', img)
    cv2.imshow('bgr', bgr_img)

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def make_pyramid(img):
    output = img.copy()
    layer = img.copy()
    while True:
        row_layer, col_layer = layer.shape[:2]
        if row_layer==1 and col_layer==1:
            break
        layer = cv2.pyrDown(layer)
        print("Size: " + str(layer.shape))
        output = concat_images(output, layer)
    cv2.imshow("pyramid", output.astype('uint8'))
    print(img.shape) # (960, 960, 3)
    print(output.shape) # (960, 1890, 3)

main_img = cv2.imread("me_original.jpg")
original_img = cv2.imread("me_blue.jpg")
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
cv2.imshow('original', original_img)
cv2.imshow('grayscale', gray_img)

# # 5 transformations:
# # 1. blurring
blurring_img(original_img)
blurring_img(gray_img)
#
# # 2. rotation
rotating_img(original_img, isRGB=True)
rotating_img(gray_img, isRGB=False)
#
# # 3.
just_ops(original_img, gray_img)
#
# # 4. affine transformation
transform_affine(original_img, isRGB=True)
transform_affine(gray_img, isRGB=False)
#
# # 5. RGB to HSB
rgb2hsb(original_img)
gray2rgb(gray_img)

make_pyramid(original_img)

cv2.waitKey(0)
