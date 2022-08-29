
import cv2
import pickle
import config as cfg

import func_utils

from pcn.models import load_model
from pcn.pcn import pcn_detect


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
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

        # extract crop face images
        crop_faces = func_utils.get_cropFaces(small_frame, faceBox_lists, cfg.cropImg_size)

        # get face encodings
        face_encodings = func_utils.get_face_encodings(crop_faces)

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
