import cv2
import os
import pickle
import numpy as np
from PIL import Image
from find_faces import find_faces

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'users')

cascade_path = 'C:/Users/Gabriel/Documents/Projects/facial_recognition/data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()

train_ids = {}
current_id = 0

y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(root)

            if label not in train_ids:
                train_ids[label] = current_id
                current_id += 1
            id_ = train_ids[label]
            print(label)
            pil_image = Image.open(path).convert('L')
            image_array = np.array(pil_image, 'uint8')
            # faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            faces = find_faces(image_array
                               )
            for (x,y,w,h) in faces:
                roi_train = image_array[y:y+h, x:x+w]
                x_train.append(roi_train)
                y_labels.append(id_)



# save label ids
pickle.dump(train_ids, open('labels.pkl', 'wb'))

recognizer.train(x_train,np.array(y_labels))
recognizer.save('trainner.yml')
