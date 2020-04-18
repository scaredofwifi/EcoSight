import csv
import numpy as np
import pandas as pd
import os
import glob
import cv2
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

# parameters
num_trees = 100
test_size = 0.10
seed = 9
train_path = "dataset/train"
test_path = "dataset/test"
scoring = "accuracy"

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))

# variables to hold the results and names
results = []
names = []

# import the feature vector and trained labels
# my_data = np.genfromtxt('dataframes/all.csv', delimiter=',')
# all_features = my_data.T[0:17]

# read in data, delete rows with null values
data = pd.read_csv('dataframes/all.csv')
prep_data = data.dropna()
# print(data.isnull().sum()) #rectangularity 170, circularity 170, angle 508

# save prepared data set
prep_data.to_csv('dataframes/all_prepared_data.csv', index=False)

features = []
labels = []

# open data set and add necessary columns to features/labels
with open('dataframes/all_prepared_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels.append(row.get('classification'))
        features.append(
            [row['aspectRatio'], row['area'], row['perimeter'], row['rectangularity'], row['circularity'],
             row['equiDiameter'], row['angle'], row['red_mean'], row['red_std'], row['green_mean'],
             row['green_std'], row['blue_mean'], row['blue_std'], row['contrast'], row['correlation'],
             row['inverse_diff_moments'], row['entropy']])

le = LabelEncoder()
target = le.fit_transform(labels)

# convert to array
all_features = np.array(features)
classification_labels = np.array(labels)

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(all_features.shape))
print("[STATUS] labels shape: {}".format(classification_labels.shape))
print("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(all_features),
                                                                                          np.array(
                                                                                              classification_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print("[STATUS] split train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))

trainDataGlobal = trainDataGlobal.astype(np.float64)
testDataGlobal = testDataGlobal.astype(np.float64)

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
# pyplot.show()

# testing the model

# fixed size for images
fixed_size = tuple((500, 500))
# create the model - Random Forests
clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):

    # feature extraction from preprocessing
    print("Processing image...")

    img1 = cv2.imread(file)
    img1 = cv2.resize(img1, fixed_size)
    #plt.imshow(img1, cmap="Greys_r")

    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # simply converting the color image to grayscale here
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # here we will smooth the edges of the resulting grayscale image so that we can
    gBlur = cv2.GaussianBlur(grayScale, (5, 5), 0)
    retVal, thresholdImg = cv2.threshold(gBlur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # so now we use a method called "morphological transformation" to close the holes of the binary image
    tempKernel = np.ones((10, 10), np.uint8)
    closedImg = cv2.morphologyEx(thresholdImg, cv2.MORPH_CLOSE, tempKernel)

    contours, _ = cv2.findContours(closedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    plottedContours = cv2.drawContours(grayScale, contours, -1, (0, 255, 0), 5)

    # FEATURE EXTRACTION
    moments = cv2.moments(cnt)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    contours_im = cv2.drawContours(closedImg, [box], 0, (255, 255, 255), 2)

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
    '''
    textures = mh.features.haralick(grayScale)
    ht_mean = textures.mean(axis=0)
    contrast = ht_mean[1]
    correlation = ht_mean[2]
    inverse_diff_moments = ht_mean[4]
    entropy = ht_mean[8]
    '''
    contrast = 0
    correlation = 0
    inverse_diff_moments = 0
    entropy = 0
    angle = 0

    print("Done processing image: " + file)

    # Concatenate features
    global_feature = np.hstack([aspectRatio, area, perimeter, rectangularity,
                                circularity, equiDiameter, angle, red_mean, red_std,
                                green_mean, green_std, blue_mean, blue_std, contrast,
                                correlation, inverse_diff_moments, entropy])
    print(global_feature)
    global_feature = global_feature.reshape(1, -1)
    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_feature = scaler.fit_transform(global_feature)

    # predict label of test image
    prediction = clf.predict(rescaled_feature.reshape(1, -1))[0]
    print(prediction)

    # show predicted label on image
    #cv2.putText(image, train_labels[prediction], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    # display the output image
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.show()

print("done")
