import numpy as np
import os
import cv2
from matplotlib import pyplot as plt


imgFile = '/112.jpg'

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

#simply converting the color image to grayscale here
grayScale = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
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

retVal, thresholdImg = cv2.threshold(gBlur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
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