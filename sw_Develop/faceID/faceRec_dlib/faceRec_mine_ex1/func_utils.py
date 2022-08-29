
import cv2
import numpy as np

import face_recognition


class face_box:
    def __init__(self, x, y, width, angle, score):
        self.x = x
        self.y = y
        self.width = width
        self.angle = angle
        self.score = score

def rotate_point(x, y, centerX, centerY, angle):
    x -= centerX
    y -= centerY
    theta = -angle * np.pi / 180
    rx = int(centerX + x * np.cos(theta) - y * np.sin(theta))
    ry = int(centerY + x * np.sin(theta) + y * np.cos(theta))
    return rx, ry

def crop_faceFrame(img, face:face_box, crop_size=200):
    x1 = face.x - 10
    y1 = face.y - 10
    x2 = face.width + face.x - 1 + 10
    y2 = face.width + face.y - 1 + 10
    centerX = (x1 + x2) // 2
    centerY = (y1 + y2) // 2
    lst = (x1, y1), (x1, y2), (x2, y2), (x2, y1)
    pointlist = [rotate_point(x, y, centerX, centerY, face.angle) for x, y in lst]
    srcTriangle = np.array([
        pointlist[0],
        pointlist[1],
        pointlist[2],
    ], dtype=np.float32)
    dstTriangle = np.array([
        (0, 0),
        (0, crop_size - 1),
        (crop_size - 1, crop_size - 1),
    ], dtype=np.float32)
    rotMat = cv2.getAffineTransform(srcTriangle, dstTriangle)
    ret = cv2.warpAffine(img, rotMat, (crop_size, crop_size))
    return ret, pointlist

def rotate_point(x, y, centerX, centerY, angle):
    x -= centerX
    y -= centerY
    theta = -angle * np.pi / 180
    rx = int(centerX + x * np.cos(theta) - y * np.sin(theta))
    ry = int(centerY + x * np.sin(theta) + y * np.cos(theta))
    return rx, ry

def draw_line(img, pointlist):
    thick = 3
    cyan = (255, 255, 0)
    blue = (255, 0, 0)
    cv2.line(img, pointlist[0], pointlist[1], cyan, thick)
    cv2.line(img, pointlist[1], pointlist[2], cyan, thick)
    cv2.line(img, pointlist[2], pointlist[3], cyan, thick)
    cv2.line(img, pointlist[3], pointlist[0], blue, thick)

def get_face_locs(face:face_box):
    x1 = face.x
    y1 = face.y
    x2 = face.width + face.x -1
    y2 = face.width + face.y -1
    centerX = (x1 + x2) // 2
    centerY = (y1 + y2) // 2
    lst = (x1, y1), (x1, y2), (x2, y2), (x2, y1)
    pointlist = [rotate_point(x, y, centerX, centerY, face.angle) for x, y in lst]
    return pointlist


def get_face_encodings(crop_faces):
    face_encodings = []
    for crop_face in crop_faces:
        # face encodings from the crop faces by "face_recognition" lib (Adam Geitgey) developed on the "dlib"
        ind_face_encoding = np.array(face_recognition.face_encodings(crop_face, known_face_locations=None, num_jitters=1, model="small")).flatten()
        face_encodings.append(ind_face_encoding)

    return face_encodings

def get_face_locations(win_lists):
    face_locations = []
    for win_list in win_lists:
        # face locations by rotated 4 edge coordinates of boxes on detected faces
        face_location = get_face_locs(win_list)
        face_locations.append(face_location)

    return face_locations

def get_upscaled_cropFaces(small_frame, faceBox_lists, sr, cropImg_size):
    # extract crop face images
    crop_faces = list(
        map(lambda face_box: crop_faceFrame(small_frame, face_box, cropImg_size), faceBox_lists))
    crop_faces = [f[0] for f in crop_faces]

    # face image upsampling
    ups_crop_faces = []
    for crop_face in crop_faces:
        ups_crop_faces.append(sr.upsample(crop_face))

    return crop_faces, ups_crop_faces

def get_cropFaces(small_frame, faceBox_lists, cropImg_size):
    # extract crop face images
    crop_faces = list(
        map(lambda face_box: crop_faceFrame(small_frame, face_box, cropImg_size), faceBox_lists))
    crop_faces = [f[0] for f in crop_faces]

    return crop_faces


def get_face_names(known_face_encodings, data, face_encodings):
    # find known face's name
    face_names = []
    for face_encoding in face_encodings:
        name = "Unknown"

        # See if the face is a match for the known face(s)
        # compare detected face encodings with known face encodings
        matches = []
        if len(face_encoding) > 0 and np.array(face_encoding).shape[0] == np.array(known_face_encodings).shape[1]:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        face_names.append(name)

    return face_names

def draw_name_faceBox(frame, face_locations, face_names, img_ratio):
    for [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], name in zip(face_locations, face_names):
        x1 *= img_ratio
        y1 *= img_ratio
        x2 *= img_ratio
        y2 *= img_ratio
        x3 *= img_ratio
        y3 *= img_ratio
        x4 *= img_ratio
        y4 *= img_ratio

        pointlist = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        # Draw a box around the face
        draw_line(frame, pointlist)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x2 - 20, y2 + 30), font, 1.0, (255, 255, 255), 2)
    return frame


