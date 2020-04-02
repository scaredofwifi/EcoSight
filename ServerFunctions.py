'''
when you are going to call the methods: readDB, createDoc, createEmptyDoc, setDocVals, or deleteDoc. The arguments for collec
and docName should be in the form of   u'collection/document name'  and any method that has the argument attrList should
contain the attrList list from preprocessing.py
'''

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import csv

cred = credentials.Certificate('./ecosight01-firebase-adminsdk-122wv-3582c284d9.json')
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()



def readDB(collec):
    Test = db.collection(collec).stream()

    # Opening a CSV file for users data
    with open('./firebase.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Writing header row of CSV file. Change it according to your needs ;)
        writer.writerow(["area", "perimeter", "thresholImg", "aspectRatio", "rectangularity", "circularity", "equiDiameter", "angle", "classification"])

        # Writing all the users data
        for i in Test:
            writer.writerow([i.get("area"), i.get("perimeter"), i.get("thresholdImg"), i.get("aspectRatio"), i.get("rectangularity"), i.get("circularity"), i.get("equiDiameter"), i.get("angle"),i.get("classification")])

def createDoc(collec, docName, attrList):
    col_ref = db.collection(collec)
    new_val = {
        u'aspectRatio': attrList[aspectRatio],
        u'area': attrList[area],
        u'perimeter': attrList[perimeter],
        u'thresholdImg': attrList[thresholdImg],
        u'rectangularity': attrList[rectangularity],
        u'circularity': attrList[circularity],
        u'equiDiameter': attrList[equiDiameter],
        u'angle': attrList[angle],
        u'classification': attrList[classification],
    }
    col_ref.document(docName).create(new_val)

def createEmptyDoc(collec, docName):
    col_ref = db.collection(collec)
    new_val = {
        u'aspectRatio': 0,
        u'area': 0,
        u'perimeter': 0,
        u'thresholdImg': 0,
        u'rectangularity': 0,
        u'circularity': 0,
        u'equiDiameter': 0,
        u'angle': 0,
        u'classification': 'empty',
    }
    col_ref.document(docName).create(new_val)

def setDocVals(collec, docName,attrList):
    doc_ref = db.collection(collec).document(docName)
    doc_ref.set({
        u'aspectRatio': attrList[aspectRatio],
        u'area': attrList[area],
        u'perimeter': attrList[perimeter],
        u'thresholdImg': attrList[thresholdImg],
        u'rectangularity': attrList[rectangularity],
        u'circularity': attrList[circularity],
        u'equiDiameter': attrList[equiDiameter],
        u'angle': attrList[angle],
        u'classification': attrList[classification],
    })

def deleteDoc(collec, docName):
    db.collection(collec).document(docName).delete()

