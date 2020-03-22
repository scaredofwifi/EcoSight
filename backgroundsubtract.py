import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# this script will do some further background subtraction for our preprocessing
# interesting backgrounds are making it passed standard cv2 thresholding and bringing a lot of noise to binary images

# following function finds the correct shape contour by taking any coordinate point of the img (default - centerpoint)
# and checks whether the current contour contains that point or not. Returns the index of the correct contour

def find_contour(cnts, img):
    contains = []
    y_ri, x_ri, _ = img.shape
    for cc in cnts:
        yn = cv2.pointPolygonTest(cc, (x_ri // 2, y_ri // 2), False)
        contains.append(yn)
    val = [contains.index(temp) for temp in contains if temp > 0]
    print(contains)
    return val[0]


def subtract_background(image: np.ndarray):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(testImg, cmap='Greys_r')
    plt.show(block=True)

    gsImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # plt.imshow(gsImg, cmap='Greys_r')

    blur = cv2.GaussianBlur(gsImg, (55, 55), 0)
    # plt.imshow(blur, cmap='Greys_r')
    # plt.show(block=True)

    retOtsu, imgOtsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plt.imshow(imgOtsu, cmap='Greys_r')
    plt.show(block=True)

    tempKernel = np.ones((5, 5), np.uint8)
    closedImg = cv2.morphologyEx(imgOtsu, cv2.MORPH_CLOSE, tempKernel)
    plt.imshow(closedImg, cmap='Greys_r')

    # contours
    contours, hierarchy = cv2.findContours(closedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    # creating a mask image for background subtraction using the contour
    x, y = closedImg.shape

    black_img = np.empty([x, y, 3], dtype=np.uint8)
    black_img.fill(0)
    plt.imshow(black_img, cmap='Greys_r')
    plt.show(block=True)

    index = find_contour(contours, img)
    cnt = contours[index]
    mask = cv2.drawContours(black_img, [cnt], 0, (255, 255, 255), -1)
    plt.imshow(mask)
    plt.show(block=True)

    maskedImg = cv2.bitwise_and(img, mask)
    whitePx = [255, 255, 255]
    blackPx = [0, 0, 0]

    finalImg = maskedImg
    h, w, channels = finalImg.shape
    for x in range(0, w):
        for y in range(0, h):
            channels_xy = finalImg[y, x]
        if all(channels_xy == blackPx):
            finalImg[y, x] = whitePx

    plt.imshow(finalImg)
    plt.show(block=True)
    return finalImg


testImgFile = '111.png'
imgFilePath =  'C://Users/Cade Norman/Desktop/School/Capstone/plant images/' + testImgFile
print(os.path.exists(imgFilePath))
testImg = cv2.imread(imgFilePath)

subtract_background(testImg)