EcoSight Image Processing
  Used to do plant identification based on an image taken by either the EcoSight mobile applciation or a developed hardware camera that 
  will be able to do some object detection. 
 
 Python Version: Python 3.7
 Virtual Environment is based in PyCharm
 
 Dependencies:
  1) OpenCV2
  2) Numpy
  3) Pandas
  4) scikitlearn
  5) matplotlib
  
 
Image Processing will be broken into multiple python scripts. 
The first of which is preprocessing.py. This will be the first part of the 
image processing pipline.
preprocessing.py simply:
  1) reads the image of interest and converts to a numpy ndarray
  2) converts the image to greyscale
  3) smoothes the edges of the image using "Gaussian Blur"
  4) uses adaptive thresholding, Otsu's Technique, to create a binary image
  5) does boundary detection using sobel filters
  
This image preprocessing will allow us to do feature detection in the next part of the image processing pipeline.
