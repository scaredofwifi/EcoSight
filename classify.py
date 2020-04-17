import scipy as sc
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

import os


def build_model():
    datasetPath = os.getcwd() + '\\dataframes\\all.csv'
    ds = pd.read_csv(datasetPath)
    print(ds.shape)
    print(ds.head(20))
    print(ds.describe())
    ds.plot(kind='box', subplots=True, sharex=False, sharey=False)
    plt.show()

build_model()