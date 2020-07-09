import cv2

def find_faces(gray_frame):
    found_faces = []
    cascade_files = ['C:/Users/Gabriel/Documents/Projects/facial_recognition/data/haarcascade_frontalface_alt.xml',
                     'C:/Users/Gabriel/Documents/Projects/facial_recognition/data/haarcascade_frontalface_default.xml',
                     'C:/Users/Gabriel/Documents/Projects/facial_recognition/data/haarcascade_frontalface_alt2.xml']
    face_cascades = [cv2.CascadeClassifier(cascade_filepath) for cascade_filepath in cascade_files]

    for cascade in face_cascades:
        faces = cascade.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)
        if len(face_cascades) > 1:
            for face in faces:
                overlapping = [False] + [is_overlapping(face, found_face) for found_face in found_faces]
                if True not in overlapping:
                    found_faces.append(face)
        else:
            found_faces = faces
    return found_faces

def is_overlapping(rec_a,rec_b):
    (a_xi,a_yi,a_w,a_h) = rec_a
    (b_xi,b_yi,b_w,b_h) = rec_b

    if (a_xi >= b_xi + b_w) or (b_xi >= a_xi + a_w):
        return False
    if (a_yi + a_h <= b_yi) or (b_yi + b_h <= a_yi):
        return False
    return True