
import cv2
import pickle
import config as cfg
from cv2 import dnn_superres
import numpy as np

import func_utils

from pcn.models import load_model
from pcn.pcn import pcn_detect


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
# image SR
# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()
# Read the desired model
path = "/home/jlee/jlee_works/proj_VISTA/py_code/faceDet_SuperResolution/image_SR_openCV/models/FSRCNN_x2.pb"
sr.readModel(path)
# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("fsrcnn", 2)

# load model for PCN
nets = load_model()

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(cfg.camID)
if (video_capture.isOpened() == False):
    print("Error opening the video streaming")
    exit()

# load the known faces and embeddings
print("Loading known faces encodings...")
data = pickle.loads(open(cfg.encodingPath, "rb").read())

# Create arrays of known face encodings and their names
known_face_encodings = data['encodings']
known_face_names = data['names']

while video_capture.isOpened():
    # Grab video frame
    ret, frame = video_capture.read()

    # Resize frame of video to 1/n size for faster face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=cfg.sImg_ratio, fy=cfg.sImg_ratio)

    # PCN face detection - return face window class on detected faces
    faceBox_lists = pcn_detect(small_frame, nets)

    if faceBox_lists:
        # get face locations
        face_locations = func_utils.get_face_locations(faceBox_lists)

        # get crop faces and upsampled crop faces
        crop_faces, ups_crop_faces = func_utils.get_upscaled_cropFaces(small_frame, faceBox_lists, sr, cfg.cropImg_size)

        # crop_faces_rs = cv2.resize(crop_faces[0], (0, 0), fx=3, fy=3)
        # cv2.imshow('corp_face', crop_faces_rs)
        # cv2.imshow('ups_corp_face', ups_crop_faces[0])

        # get face encodings
        face_encodings = func_utils.get_face_encodings(ups_crop_faces)

        # find known face's name
        face_names = func_utils.get_face_names(known_face_encodings, data, face_encodings)

        # Display the results
        frame = func_utils.draw_name_faceBox(frame, face_locations, face_names, cfg.lImg_ratio)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
