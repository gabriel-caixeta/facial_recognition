import numpy as np
import cv2
import datetime
import os
from find_faces import find_faces

TIME_INTERVAL = 3

cascade_files = ['C:/Users/Gabriel/Documents/Projects/facial_recognition/data/haarcascade_frontalface_alt.xml',
                 'C:/Users/Gabriel/Documents/Projects/facial_recognition/data/haarcascade_frontalface_default.xml',
                 'C:/Users/Gabriel/Documents/Projects/facial_recognition/data/haarcascade_frontalface_alt2.xml']

cascade_path = 'C:/Users/Gabriel/Documents/Projects/facial_recognition/data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)

username = 'kacey'

user_folder = f'users/{username}'
if not os.path.exists(user_folder):
    os.makedirs(user_folder)

current_files = []
for root, dir, files in os.walk(user_folder):
    current_files.extend(files)

last_photo = datetime.datetime.now()



while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find faces
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    faces = find_faces(gray)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]

        if datetime.datetime.now() > last_photo + datetime.timedelta(seconds=TIME_INTERVAL):
            img_filename = f'{user_folder}/{str(len(current_files))}.png'
            cv2.imwrite(img_filename, frame)
            print(f'Added data {img_filename} to {username}')
            last_photo = datetime.datetime.now()
            current_files.append(img_filename)

        color = (255, 0, 0)
        stroke = 2
        cv2.rectangle(frame, (x, y), (x+w,y+h), color, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()