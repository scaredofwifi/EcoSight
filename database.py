'''
*******************************************************************************************************************************************************************
All functions excluding upload_file and download_file
*******************************************************************************************************************************************************************

when you are going to call the methods: read_db, read_doc, create_doc, create_empty_doc, set_doc_vals, or delete_doc
The arguments for collec and docName should be in the form of:
    'collec' or 'docName'

Any method that has the argument attrList should contain the attrList list from preprocessing.py

When reading, note that it will export a csv to the directory where the file you have open is, additionally every csv that comes from the read_db,
reading an entire collection, or read_doc, reading a document from a collection, will have a name of:
    firebase_collection.csv and firebase_document.csv respecitvely


*******************************************************************************************************************************************************************
upload_file and download_file
*******************************************************************************************************************************************************************

When uploading or downloading files, the file_name argument must have the correct file name with the dot extension and must be in single quotes:
    'download.jpg' would be acceptble.

For the file_path argument, it varies. When uploading, like the file_name, nust be in single quotes, and must containg the file name and extension
on the end of the path. Every slash must be a double slash:
    'C:\\Users\\doodl\\Documents\\PyCharm\\ServerFunctions\\download.jpg', would be acceptable if uploading the download.jpg file

For Downloading it is the same file_path as before except without the file on the end of the path:
    'C:\\Users\\doodl\\Documents\\PyCharm\\ServerFunctions' would be acceptable

***IMOPRTANT***
When downloading the file and proving the path, please put the \\ after the directory you wish it to be in otherwise it will download into the parent directory:
    'C:\\Users\\doodl\\Documents\\PyCharm\\ServerFunctions\\' would save into ServerFunctions, whereas:
    'C:\\Users\\doodl\\Documents\\PyCharm\\ServerFunctions' would save as ServerFunctionsfile_name in Pycharm

For uploading files, checking if a file exits seems to be broken so there is no safeguard and will crash the program with an error. Be very careful about your naming
and paths.


*******************************************************************************************************************************************************************
Forebase Firestore Document Layout Visualized
*******************************************************************************************************************************************************************

the hierarchy exists as such collection->documents, in documents, data is held


*******************************************************************************************************************************************************************
Collections And Check If Existing Documents or Collections Disclaimer
*******************************************************************************************************************************************************************

*** IMPORTANT ***
I have already premade a collection labeled as family, called ina function as 'family', which is all you should need. However, if you for some reason feel inclined
to make another collection, simply call either of the create document functions and put a new name in the collec argument and then of course include a docName.
Additionally, if you end up deleting all documents from a collection the collection will disappear, thus you should always have an empty document called
    collec_holder
In your collection at all times.

Unfortunately, there doesn't seem to be any reliable way to check if documents or collections exist, so try to just try to enter the names of eveything correctly.
Even if you don't enter them correctly, don't worry no errors will be thrown, it'll still export a csv file with empty variables so no errors should be thrown.
The same is said for every other function aside from upload and download, they will all not thow errors, but will likely instead create a new document or colleciton.

*******************************************************************************************************************************************************************
END
*******************************************************************************************************************************************************************
'''

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import pandas as pd
import csv
import urllib.request
import datetime
from datetime import timedelta

cred = credentials.Certificate('./ecosight01-firebase-adminsdk-122wv-3582c284d9.json')
bucket_app = firebase_admin.initialize_app(cred, {'storageBucket': 'ecosight01.appspot.com'})
db = firestore.client()
bucket = storage.bucket()



def read_db(collec):
    Test = db.collection(collec).stream()
    # Opening a CSV file for users data
    with open('./firebase_collection.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Writing header row of CSV file. Change it according to your needs ;)
        writer.writerow(["area", "perimeter", "thresholImg", "aspectRatio", "rectangularity", "circularity", "equiDiameter", "angle", "classification"])

        # Writing all the users data
        for i in Test:
            writer.writerow([i.get("area"), i.get("perimeter"), i.get("thresholdImg"), i.get("aspectRatio"), i.get("rectangularity"), i.get("circularity"), i.get("equiDiameter"), i.get("angle"),i.get("classification")])

def read_doc(collec, docName):
    Test = db.collection(collec).document(docName).get()
    # Opening a CSV file for users data
    with open('./firebase_document.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Writing header row of CSV file. Change it according to your needs ;)
        writer.writerow(["area", "perimeter", "thresholImg", "aspectRatio", "rectangularity", "circularity", "equiDiameter", "angle", "classification"])
        writer.writerow([Test.get("area"), Test.get("perimeter"), Test.get("thresholdImg"), Test.get("aspectRatio"), Test.get("rectangularity"), Test.get("circularity"), Test.get("equiDiameter"), Test.get("angle"),Test.get("classification")])

def create_doc(collec, docName, attrList):
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

def create_empty_doc(collec, docName):
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

def set_doc_vals(collec, docName,attrList):
    try:
        doc_ref = db.collection(collec).document(docName)
    except:
        print("The Document "+ docName+" Does Not Exist")
        return

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

def set_doc_vals_empty(collec, docName):
    doc_ref = db.collection(collec).document(docName)
    doc_ref.set({
        u'aspectRatio': 0,
        u'area': 0,
        u'perimeter': 0,
        u'thresholdImg': 0,
        u'rectangularity': 0,
        u'circularity': 0,
        u'equiDiameter': 0,
        u'angle': 0,
        u'classification': 'empty',
    })
def delete_doc(collec, docName):
    db.collection(collec).document(docName).delete()

def upload_file(file_name, local_path):
    blob = bucket.blob(file_name)
    outfile = local_path
    with open(outfile, 'rb') as my_file:
        blob.upload_from_file(my_file)

def download_file(file_name, local_path):
    try:
        blob = bucket.blob(file_name)
        url = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        full_path = local_path + file_name
        urllib.request.urlretrieve(url, full_path)
    except:
        print("File Name Not Found: " + file_name)
        return


