import numpy as np
import cv2
import os

import FaceTest as fr


def detectFace(test_img):

    face_detected, grayImage = fr.faceDetection(test_img)

    print("Face detected : ", face_detected)

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



    return test_img