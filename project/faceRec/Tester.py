####################################################################################
# Author:- MANAS KUMAR MISHRA
# Task:- Face recognition
# DIP project 
####################################################################################


# Import the various libraries
import numpy as np
import cv2
import os

import FaceTest as fr

test_img = cv2.imread('/home/manas/Documents/ImageProcessing/project/faceRec/UnknownFace/un2.jpeg')


face_detected, grayImage = fr.faceDetection(test_img)

print("Face detected : ", face_detected)



# for (x, y, w, h) in face_detected:
#     cv2.rectangle(test_img, (x,y),(x+w, y+h), (0, 255, 0), thickness = 5)


# resized_img = cv2.resize(test_img, (1000, 700))

# cv2.imshow("Face detection", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# training of the model 
# faces, faceID = fr.lebels_for_training_data("/home/manas/Documents/ImageProcessing/project/faceRec/KnownFace")

# faceReconizer = fr.train_classifier(faces, faceID)

# faceReconizer.save('trainingData.yml') 

faceReconizer = cv2.face.LBPHFaceRecognizer_create()

faceReconizer.read("/home/manas/Documents/ImageProcessing/project/faceRec/trainingData.yml")

name = {0:"MANAS",
        1: "VIJAY",
        2:"Sachine"}


for face in face_detected:
    (x,y,w,h) = face

    roi_gray = grayImage[y:y+h, x:x+h]

    label, confidence = faceReconizer.predict(roi_gray)

    print("Confodence : ", confidence)
    print("Probability of correct :", 100-confidence)
    print("Label :", label)
    print("Predicted name : ", name[label])

    fr.draw_rect(test_img, face)

    predicted_name=name[label]


    if confidence> 70:
        continue

    fr.put_text(test_img, predicted_name , x, y)



resized_img = cv2.resize(test_img, (1000, 700))

cv2.imshow("Face detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()