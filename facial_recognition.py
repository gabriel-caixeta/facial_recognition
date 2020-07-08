import numpy as np
import cv2
import pickle


def rectangle(frame,pos_start,pos_end):
    color=(255,0,0)
    stroke=2
    cv2.rectangle(frame,pos_start,pos_end,color,stroke)


def text(frame,pos,content):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    stroke = 2
    cv2.putText(frame, content, pos, font, 1, color, stroke, cv2.LINE_AA, )


class FacialRecognition:
    def __init__(self, video=None):
        cascade_path = 'C:/Users/Gabriel/Documents/Projects/facial_recognition/data/haarcascade_frontalface_alt2.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainner.yml')

        label_ids = pickle.load(open('labels.pkl','rb'))
        self.label_ids = {v:k for k,v in label_ids.items()}

        self.cap = cv2.VideoCapture(video) if video else cv2.VideoCapture(0)


    def run(self):
        while(True):
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # find faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi_gray = gray[y:y+h,x:x+w]

                # recognize
                id_, conf = self.recognizer.predict(roi_gray)
                # print(self.recognizer.predict(roi_gray))
                if 45 < conf < 90:
                    print(id_, conf)
                    name = self.label_ids[id_]
                    text(frame, (x,y), name)

                rectangle(frame, (x,y), (x+w,y+h))
            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        self.terminate()

    def terminate(self):
        print('Terminating')
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Initializing')
    fr = FacialRecognition()
    fr.run()
