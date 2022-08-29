import dlib
import scipy.misc
import numpy as np
import os
import cv2
from imutils import paths
import face_recognition
import argparse
import pickle
import config as cfg

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(cfg.knownFacePath))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
face_encodings = []# Loop over images to get the encoding one by one

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb, model=cfg.faceDetModel)
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encoding)
        knownNames.append(name)


    # image_face = face_recognition.load_image_file(imagePath)
    # face_encoding = face_recognition.face_encodings(image_face)[0]

    # knownEncodings.append(face_encoding)
    # knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(cfg.encodingPath, "wb")
f.write(pickle.dumps(data))
f.close()




