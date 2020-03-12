import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from EcoSightDFM import dataFrameManager as dfm

def imgpreprocessing(imgFile: str, classification: str) -> list:
    imgFilePath = 'C://Users/Cade Norman/Desktop/School/Capstone/plant images' + imgFile

    # some debugging to check that the file path to the image of interest exists
    print(os.path.exists(imgFilePath))

    mesImg = cv2.imread(imgFilePath)
    img = cv2.cvtColor(mesImg, cv2.COLOR_BGR2RGB)

    plt.imshow(mesImg)
    plt.show(block=True)

    # turning on matplotlib's interactive mode. Not sure why its necessary, but it is.
    plt.interactive(True)

    # in order to use matplotlib's imshow(), you have to follow the call with a seperate show() call
    # the parameter of show() must be "block=True" of False. True means that you want the script to halt its
    # execution until you exit out of the window, false means the inverse. Here is an example:
    # plt.imshow(img)
    # plt.show(block=True)

    # simply converting the color image to grayscale here
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.imshow(grayScale, cmap='Greys_r')
    # cmap="Grays_r" is necessary for matplotlib to show grayscale
    plt.show(block=True)
    # getting size of the image
    print(grayScale.shape)

    # here we will smooth the edges of the resulting grayscale image so that we can
    # get a more define edge to analyse. The method for doing so is something called "Gaussian Blur"
    gBlur = cv2.GaussianBlur(grayScale, (5, 5), 0)
    # the ksize (5,5) argument corresponds to a "Kernel Size"
    # the ksize is a matrix of which you can define the "convulution mask"
    # basically the larger the values of the matrix, the more blurry the whole image is
    # but also the more the edges are smoothed. This value will probably have to be played with
    # started out with (25,25) but the resulting image was way too blurry.

    plt.imshow(gBlur, cmap='Greys_r')
    plt.show(block=True)

    # do image thresholding using Otsu's method. This simply sets an intensity threshold
    # and checks every pixel. If the pixel is above the threshold, the pixel is set to black
    # if the pixel is below the threshold, it is set to white. This allows us to distinguish between
    # the foreground and background of the image and create a "binary image".

    retVal, thresholdImg = cv2.threshold(gBlur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plt.imshow(thresholdImg, cmap='Greys_r')
    plt.show(block=True)

    # so now we use a method called "morphological transformation" to close the holes of the binary image
    tempKernel = np.ones((10, 10), np.uint8)
    # this kernel value is pretty touchy. Not sure what the best method for determining
    # the best kernel values for theses images might be. it will depend on the lighting, size and saturation of the image
    # TODO: Figure out a method based on the items listed above to determine optimal kernel values for necessary operations
    closedImg = cv2.morphologyEx(thresholdImg, cv2.MORPH_CLOSE, tempKernel)
    plt.imshow(closedImg, cmap='Greys_r')
    plt.show(block=True)

    # Now we try to extract the boundary of the image from the filled in binary image using "sobel filters"
    sobelx64f = cv2.Sobel(closedImg, cv2.CV_64F, 1, 0, ksize=5)
    # once again the kernel size is up in the air. Most documentation in showing 5 so thats what I am going with
    # calculating the absolute value of our ndarry here, without it things were not looking good. Kind of a bandaid fix
    abs_sobelx64f = np.absolute(sobelx64f)
    # converting into uint8 so that we can do a more simplified thresholding later
    sobel_8u = np.uint8(abs_sobelx64f)
    plt.imshow(abs_sobelx64f, cmap='Greys_r')
    plt.show(block=True)
    # getting decent results off sobel filters but going to try and use another type of morphological transformation
    # called Dilation. Instead of closing holes like we did earlier with morph trans, we will try and expand the width of
    # our boundary that we just extracted to make it more defined and noticable. Giving us more data to work with.
    # this probably won't be optimal for all images after using sobel filters but will be for some
    # TODO: Create method to determine whether or not to send the image through another round of MorphTrans (Dialation)
    dilationImg = cv2.dilate(abs_sobelx64f, (10, 10), 2)
    plt.imshow(dilationImg, cmap='Greys_r')
    plt.show(block=True)
    # this is not very effective so maybe we need to just not do it at all. Played with kernel values and it did not do much
    # gonna run with it for now though because it does make a bit of difference. Later on when we are developing feature
    # extraction we can see if it is just a waste of time in terms of its effects on accuracy

    retSobel, binSobel = cv2.threshold(sobel_8u, 1, 255, cv2.THRESH_BINARY)
    plt.imshow(binSobel, cmap='Greys_r')
    plt.show(block=True)
    # this was a far more effective method in terms of defining the outline better, I just ran it through another
    # binary threshold iteration but instead of it being inverted, it is standard.

    # going to try and close the edges with ONE MORE morphological transformation so that the edges are cleaner
    tempKernel2 = np.ones((10, 5), np.uint8)
    closedEdgesSobelBin = cv2.morphologyEx(binSobel, cv2.MORPH_CLOSE, tempKernel2)
    plt.imshow(closedEdgesSobelBin, cmap='Greys_r')
    plt.show(block=True)
    # not getting a clean border without any holes with any kernel value in tempKernel2
    # going to try another method of border extraction using contours

    # going to try some contours stuff here
    contours, hierarchy, = cv2.findContours(closedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    cnt = contours[0]
    print(len(cnt))

    plottedContours = cv2.drawContours(grayScale, contours, -1, (0, 255, 0), 5)
    plt.imshow(plottedContours, cmap='Greys_r')
    plt.show(block=True)

    # FEATURE EXTRACTION
    # Here we calculate the "image moments" which is basically the center of a "blob"
    # a blob is a group of adjacent pixels that share a common property, i.e. grayscale value
    # so by calculating the center of these blobs, we can identify features of the image

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
        dfm.appendtodataframe(dataframename, attributes)
        return
    else:
        dfm.createnewdataframe(dataframename=dataframename)
        attributes = imgpreprocessing(imgFile, classification)
        dfm.appendtodataframe(dataframename=dataframename, values=attributes)
        return