# WHY? :- Because, I want to test the DIP algorithms to recognize the face. Before, doing that, I have to detect the face in the webcam

import cv2
import sys

# cascPath = sys.argv[0]

faceCascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_frontalface_default.xml')


video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE 
    )

    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break




video_capture.release()

cv2.destroyAllWindows()