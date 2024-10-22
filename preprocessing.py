import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from dataframeManager import DataFrameManager
import mahotas as mh
import backgroundsubtract as bgsub




def imgpreprocessing(imgFile: str, classification: str) -> list:
     print("Processing image: " + imgFile)

     img1 = cv2.imread(imgFile)
     #plt.imshow(img1, cmap="Greys_r")

     #plt.show(block=True)
     # img1, cnt = bgsub.subtract_background(tempimg)
     #plt.imshow(img1, cmap='Greys_r')
     #plt.show(block=True)

     img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
     # turning on matplotlib's interactive mode. Not sure why its necessary, but it is.
     #plt.interactive(True)
     # simply converting the color image to grayscale here
     grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
     #plt.imshow(grayScale, cmap='Greys_r')
     # cmap="Grays_r" is necessary for matplotlib to show grayscale
     #plt.show(block=True)
     # getting size of the image
     #print(grayScale.shape)
     # here we will smooth the edges of the resulting grayscale image so that we can
     gBlur = cv2.GaussianBlur(grayScale, (5, 5), 0)
     # the ksize (5,5) argument corresponds to a "Kernel Size"
     #plt.imshow(gBlur, cmap='Greys_r')
     #plt.show(block=True)
     #do image thresholding using Otsu's method.
     retVal, thresholdImg = cv2.threshold(gBlur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
     # print("Thresholding image")
     #plt.imshow(thresholdImg, cmap='Greys_r')
     #plt.show(block=True)

    # so now we use a method called "morphological transformation" to close the holes of the binary image
     tempKernel = np.ones((10, 10), np.uint8)
    # TODO: Figure out a method based on the items listed above to determine optimal kernel values for necessary operations
     closedImg = cv2.morphologyEx(thresholdImg, cv2.MORPH_CLOSE, tempKernel)
     #plt.imshow(closedImg, cmap='Greys_r')
     #plt.show(block=True)
     '''
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
'''
     contours, _ = cv2.findContours(closedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     # print(len(contours))

     cnt = contours[0]
     # print(len(cnt))

     plottedContours = cv2.drawContours(grayScale, contours, -1, (0, 255, 0), 5)
     #plt.imshow(plottedContours, cmap='Greys_r')
     #plt.show(block=True)

    # FEATURE EXTRACTION
     moments = cv2.moments(cnt)
     # print(moments)

     area = cv2.contourArea(cnt)
     # print(area)

     perimeter = cv2.arcLength(cnt, True)
     # print(perimeter)

     rect = cv2.minAreaRect(cnt)
     box = cv2.boxPoints(rect)
     box = np.int0(box)
     contours_im = cv2.drawContours(closedImg, [box], 0, (255, 255, 255), 2)
     # plt.imshow(contours_im, cmap='Greys_r')
     # plt.show(block=True)

     try:
          ellipse = cv2.fitEllipse(cnt)
          im = cv2.ellipse(closedImg, ellipse, (255, 255, 255), 2)
          # plt.imshow(closedImg, cmap="Greys_r")
     except:
          print("Fitting an ellipse failed")


     x, y, w, h = cv2.boundingRect(cnt)
     try:
          aspectRatio = float(w) / h
          # print(aspectRatio)
     except:
          aspectRatio = 'null'

     try:
          rectangularity = w * h / area
          # print(rectangularity)
     except:
          rectangularity = 'null'

     try:
          circularity = (perimeter ** 2 / area)
          # print(circularity)
     except:
          circularity = 'null'

     try:
          equiDiameter = np.sqrt(4 * area / np.pi)
          # print(equiDiameter)
     except:
          equiDiameter = 'null'
     try:
          (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
     except:
          print("Could not calculate ellipse angle")
          angle = 'null'

     # Color features
     red_channel = img1[:, :, 0]
     green_channel = img1[:, :, 1]
     blue_channel = img1[:, :, 2]
     blue_channel[blue_channel == 255] = 0
     green_channel[green_channel == 255] = 0
     red_channel[red_channel == 255] = 0

     red_mean = np.mean(red_channel)
     green_mean = np.mean(green_channel)
     blue_mean = np.mean(blue_channel)

     red_std = np.std(red_channel)
     green_std = np.std(green_channel)
     blue_std = np.std(blue_channel)

     # Texture features
     textures = mh.features.haralick(grayScale)
     ht_mean = textures.mean(axis=0)
     contrast = ht_mean[1]
     correlation = ht_mean[2]
     inverse_diff_moments = ht_mean[4]
     entropy = ht_mean[8]

     # find out if we are trying to classify or generate a dataframe
     # if classification argument is UNKNOWN we forego appending classification to the attrList
     if classification == 'UNKNOWN':
          attrList = list([aspectRatio, area, perimeter, rectangularity,
                           circularity, equiDiameter, angle, red_mean, red_std,
                           green_mean, green_std, blue_mean, blue_std, contrast,
                           correlation, inverse_diff_moments, entropy])
     else:
          attrList = list([aspectRatio, area, perimeter, rectangularity,
                           circularity, equiDiameter, angle, red_mean, red_std,
                           green_mean, green_std, blue_mean, blue_std, contrast,
                           correlation, inverse_diff_moments, entropy, classification])


     print("Done processing image: " + imgFile + '\n Attribute List: ' + str(attrList))
     return attrList

def generate_dataframes():

     #####################################################
     # METHOD USED TO GENERATE DATAFRAMES OF EXTRACTED   #
     # FEATURES FROM DATASET TO TRAIN CLASSIFICATION MODEL#
     ######################################################

     datasetPath = os.getcwd() + '\\dataset\\dataset\\train\\'
     dsPath = os.getcwd() + "/"
     dfm = DataFrameManager()

     print("File path: " + datasetPath)
     print(os.path.exists(datasetPath))

     dfm.create_new_df(dataframename='all')
     for directory in os.listdir(datasetPath):
          dfm.create_new_df(dataframename=directory)
          newPath = datasetPath + directory
          for file in os.listdir(newPath):
               tempPath = newPath + "\\" + file
               attrList = imgpreprocessing(tempPath, directory)
               dfm.append_to_df(dataframename=directory, data=attrList)
               dfm.append_to_df(dataframename='all', data=attrList)
          dfm.print_df(dataframename=directory)
          dfm.export_df(dataframename=directory)

     dfm.export_df(dataframename='all')


