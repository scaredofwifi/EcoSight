import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# this script will do some further background subtraction for our preprocessing
# interesting backgrounds are making it passed standard cv2 thresholding and bringing a lot of noise to binary images

testImgFile = '111.png'
imgFilePath =  'C://Users/Cade Norman/Desktop/School/Capstone/plant images/' + testImgFile
print(os.path.exists(imgFilePath))
testImg = cv2.imread(imgFilePath)
img = cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB)
plt.imshow(testImg, cmap='Greys_r')
plt.show(block=True)

gsImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gsImg, cmap='Greys_r')

blur = cv2.GaussianBlur(gsImg, (55, 55), 0)
plt.imshow(blur, cmap='Greys_r')
plt.show(block=True)

retOtsu, imgOtsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.imshow(imgOtsu, cmap='Greys_r')
plt.show(block=True)

tempKernel = np.ones((50, 50), np.uint8)
closedImg = cv2.morphologyEx(imgOtsu, cv2.MORPH_CLOSE, tempKernel)
plt.imshow(closedImg, cmap='Greys_r')

#contours
contours, hierarchy = cv2.findContours(closedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

# following function finds the correct shape contour by taking any coordinate point of the img (default - centerpoint)
# and checks whether the current contour contains that point or not. Returns the index of the correct contour

def find_contour(cnts):
    contains = []
    y_ri, x_ri, _ = img.shape
    for cc in cnts:
        yn = cv2.pointPolygonTest(cc, (x_ri // 2, y_ri // 2), False)
        contains.append(yn)
    val = [contains.index(temp) for temp in contains if temp > 0]
    print(contains)
    return val[0]

# creating a mask image for background subtraction using the contour
black_img = np.empty([1200, 1600, 3], dtype=np.uint8)
black_img.fill(0)
plt.imshow(black_img, cmap='Greys_r')
plt.show(block=True)

index = find_contour(contours)
cnt = contours[index]
mask = cv2.drawContours(black_img, [cnt], 0, (255, 255, 255), -1)
plt.imshow(mask)
plt.show(block=True)

