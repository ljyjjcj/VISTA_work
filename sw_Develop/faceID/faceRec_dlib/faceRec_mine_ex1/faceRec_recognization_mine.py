import face_recognition
import cv2
import numpy as np
import pickle
import config as cfg

# ------------------------------------------------------------------
# TODO
# install cuDNN and use face detection model by CNN
# ------------------------------------------------------------------

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

img_red_ratio = 2
s_img_ratio = 1/img_red_ratio

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(cfg.camID)


# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(cfg.encodingPath, "rb").read())

# Create arrays of known face encodings and their names
known_face_encodings = data['encodings']
known_face_names = data['names']

# Initialize some variables
face_locations = []
face_encodings = []

while video_capture.isOpened():
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=s_img_ratio, fy=s_img_ratio)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    boxes = face_recognition.face_locations(rgb_small_frame, model=cfg.faceDetModel)
    encodings = face_recognition.face_encodings(rgb_small_frame, boxes)

    # initialize the list of names for each face detected
    names = []
    face_names = []
    for face_encoding in face_encodings:
        
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        print(known_face_encodings)
        print(face_encoding)
        print(face_encoding[0])
        print(face_encoding[0].shape)
        input()
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        face_names.append(name)


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= img_red_ratio
        right *= img_red_ratio
        bottom *= img_red_ratio
        left *= img_red_ratio

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    #cv2.imwrite('faster_rec_webcam.jpg', frame)
    # print(name)
    # cv2.imwrite(f"output_images/{name}_faster_recog.jpg", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
