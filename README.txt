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

EcoSightDFM.py 

Acts as a way for us to manage all data gained from the image processing. It creates an API, in the form of a "Dataframe Manager", 
designed for more convienient interactions with pandas and our specific use of it. It makes sure that image data is partitioned into 
certain classifications an allows image processing data to be easier to import into a pandas data frame. Pandas makes csv manipulation 
and decision tree algorithms simple. 

The highlights of the dmf API are:

  createnewdataframe(dataframename=x)
      creates an empty and new image classification dataframe. Key is designated by the dataframename
      parameter
  
  appendtodataframe(dataframename=x, values=y[])
      allows you to add a list of image processing information to a dataframe which is accessed by
      the key given in first parameter
  
  printdataframes() 
      allows you to see all dataframes created and active in the dataframe manager
      
