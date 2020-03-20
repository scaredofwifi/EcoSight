import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from EcoSightDFM import DataFrameManager

global dfm
dfm = DataFrameManager()
path = 'C://Users/Cade Norman/Desktop/School/Capstone/plant images'

def imgpreprocessing(imgFile: str, classification: str) -> list:
     imgFilePath =  'C://Users/Cade Norman/Desktop/School/Capstone/plant images/' + imgFile
     print(os.path.exists(imgFilePath))
     mesImg = cv2.imread(imgFilePath)
     img = cv2.cvtColor(mesImg, cv2.COLOR_BGR2RGB)
     plt.imshow(mesImg)
     plt.show(block=True)
     # turning on matplotlib's interactive mode. Not sure why its necessary, but it is.
     plt.interactive(True)
     # simply converting the color image to grayscale here
     grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
     plt.imshow(grayScale, cmap='Greys_r')
     # cmap="Grays_r" is necessary for matplotlib to show grayscale
     # plt.show(block=True)
     # getting size of the image
     print(grayScale.shape)
     # here we will smooth the edges of the resulting grayscale image so that we can
     gBlur = cv2.GaussianBlur(grayScale, (5, 5), 0)
     # the ksize (5,5) argument corresponds to a "Kernel Size"
     plt.imshow(gBlur, cmap='Greys_r')
     plt.show(block=True)
     #do image thresholding using Otsu's method.
     retVal, thresholdImg = cv2.threshold(gBlur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
     plt.imshow(thresholdImg, cmap='Greys_r')
     plt.show(block=True)

    # so now we use a method called "morphological transformation" to close the holes of the binary image
     tempKernel = np.ones((10, 10), np.uint8)
    # TODO: Figure out a method based on the items listed above to determine optimal kernel values for necessary operations
     closedImg = cv2.morphologyEx(thresholdImg, cv2.MORPH_CLOSE, tempKernel)
     plt.imshow(closedImg, cmap='Greys_r')
     plt.show(block=True)

    # Now we try to extract the boundary of the image from the filled in binary image using "sobel filters"
     sobelx64f = cv2.Sobel(closedImg, cv2.CV_64F, 1, 0, ksize=5)

     abs_sobelx64f = np.absolute(sobelx64f)
     # converting into uint8 so that we can do a more simplified thresholding later
     sobel_8u = np.uint8(abs_sobelx64f)
     plt.imshow(abs_sobelx64f, cmap='Greys_r')
     plt.show(block=True)

    # TODO: Create method to determine whether or not to send the image through another round of MorphTrans (Dialation)
     dilationImg = cv2.dilate(abs_sobelx64f, (10, 10), 2)
     plt.imshow(dilationImg, cmap='Greys_r')
     plt.show(block=True)

     retSobel, binSobel = cv2.threshold(sobel_8u, 1, 255, cv2.THRESH_BINARY)
     plt.imshow(binSobel, cmap='Greys_r')
     plt.show(block=True)

     tempKernel2 = np.ones((10, 5), np.uint8)
     closedEdgesSobelBin = cv2.morphologyEx(binSobel, cv2.MORPH_CLOSE, tempKernel2)
     plt.imshow(closedEdgesSobelBin, cmap='Greys_r')
     plt.show(block=True)

     contours, hierarchy, = cv2.findContours(closedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     print(len(contours))

     cnt = contours[0]
     print(len(cnt))

     plottedContours = cv2.drawContours(grayScale, contours, -1, (0, 255, 0), 5)
     plt.imshow(plottedContours, cmap='Greys_r')
     plt.show(block=True)

    # FEATURE EXTRACTION
     moments = cv2.moments(cnt)
     print(moments)

     area = cv2.contourArea(cnt)
     print(area)

     perimeter = cv2.arcLength(cnt, True)
     print(perimeter)

     rect = cv2.minAreaRect(cnt)
     box = cv2.boxPoints(rect)
     box = np.int0(box)
     contours_im = cv2.drawContours(closedImg, [box], 0, (255, 255, 255), 2)
     plt.imshow(contours_im, cmap='Greys_r')
     plt.show(block=True)

     ellipse = cv2.fitEllipse(cnt)
     im = cv2.ellipse(closedImg, ellipse, (255, 255, 255), 2)
     plt.imshow(closedImg, cmap="Greys_r")

     x, y, w, h = cv2.boundingRect(cnt)
     aspectRatio = float(w) / h
     print(aspectRatio)

     rectangularity = w * h / area
     print(rectangularity)

     circularity = (perimeter ** 2 / area)
     print(circularity)

     equiDiameter = np.sqrt(4 * area / np.pi)
     print(equiDiameter)

     (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
     attrList = list([img, grayScale, gBlur, thresholdImg, aspectRatio, rectangularity, circularity, classification])
     return attrList


def addattributestodf(dataframename: str, imgFile: str, classification: str):
    if dfm.hasdf(dataframename=dataframename):
        attributes = imgpreprocessing(imgFile, classification)
        dfm.appendtodataframe(dataframename=dataframename, values=attributes)
        return
    else:
        dfm.createnewdataframe(dataframename=dataframename)
        attributes = imgpreprocessing(imgFile, classification)
        dfm.appendtodataframe(dataframename=dataframename, values=attributes)
        return

for file in os.listdir(path):
     print(file)
     addattributestodf('mesquiteDf', file, 'Mesquite')