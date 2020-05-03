# TRAINING OUR MODEL
import numpy as np
import pandas as pd
import os
import glob
import cv2
import warnings
import time
import mahotas as mh
from matplotlib import pyplot
import preprocessing as pp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# read in data, delete rows with null values
data = pd.read_csv('dataframes/all.csv')
prep_data = data.dropna()
# print(data.isnull().sum()) #rectangularity 170, circularity 170, angle 508

# save prepared data set
prep_data.to_csv('dataframes/all_prepared_data.csv', index=False)

# tunable-parameters
num_trees = 175
test_size = 0.10
seed = 42
train_path = "dataset/train"
test_path = "dataset/test"
csv_data = pd.read_csv("dataframes/all_prepared_data.csv",
                       usecols=["aspectRatio", "area", "perimeter", "rectangularity",
                                "circularity", "equiDiameter", "angle", "red_mean", "red_std",
                                "green_mean", "green_std", "blue_mean", "blue_std", "contrast",
                                "correlation", "inverse_diff_moments", "entropy"])
csv_labels = pd.read_csv("dataframes/all_prepared_data.csv", usecols=["classification"])
scoring = "accuracy"

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

# create all the machine learning models
models = [('CART', DecisionTreeClassifier(random_state=seed)),
          ('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed))]

# variables to hold the results and names
results = []
names = []

global_features = np.array(csv_data)
global_labels = np.array(csv_labels)

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print("[STATUS] splitting train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))

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



# TESTING OUR MODEL


fixed_size = tuple((500, 500))

# create the model - Random Forests
clfRF = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

# fit the training data to the model
clfRF.fit(trainDataGlobal, trainLabelsGlobal)

# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):

    features = pp.imgpreprocessing(file, 'UNKNOWN')
    # Concatenate features
    global_feature = np.hstack(features)

    # global_feature = global_feature.reshape(1, -1)
    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # rescaled_feature = scaler.fit_transform(global_feature)

    # predict label of test image
    prediction = clfRF.predict(global_feature.reshape(1, -1))[0]
    print(prediction)
    # show predicted label on image
    #cv2.putText(image, train_labels[prediction], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    # display the output image
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.show()

print("done!")


class Demo:

    p_s = ['daffodil', 'daisy', 'pansy', 'sunflower']

    def __init__(self):
        self.iter = 0

    def __del__(self):
        self.iter = 0

    def demo_classify(self, iter: int) -> str:
        time.sleep(4)
        ret = self.dem_list[iter]
        self.iter = self.iter + 1
        return ret
