####################################################################################
# Author:- MANAS KUMAR MISHRA
# Task:- Face recognition
# DIP project 
####################################################################################

# Import the various libraries
import numpy as np
import cv2
import os



def faceDetection(test_img):

    # Covert the cl=olor image into gray image 
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)


    # Harr classifier is type of classifier for face 
    face_haar_cascade = cv2.CascadeClassifier('/home/manas/Documents/ImageProcessing/project/faceRec/haarcascade_frontalface_default.xml')

    # scaleFactor is for reduce the size of image such that more chance of detection 
    # minNeighbors is for prevent false positive. At least that much number of neighbour should possess
    # to be true positive.  
    faces = face_haar_cascade.detectMultiScale(
        gray_img, 
        scaleFactor=1.2, 
        minNeighbors= 3
    )

    return faces, gray_img


def lebels_for_training_data(directory):
    # Faces in the images 
    faces  = []

    # Labels related to the images 
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if(filename.startswith(".")):
                print("Skipping system file... Not an image...")

                continue
            
            # Extract the information of the label
            # It will go recursively in all directories
            id = os.path.basename(path)

            img_path = os.path.join(path, filename)

            print("Image path :", img_path)
            print("Id :", id)

            # Read the image from the path 
            test_img = cv2.imread(img_path)

            # Check the image is loaded properly or not 
            if test_img is None:
                print("this image is not loaded properly !!!")
                continue


            # Apply haar classifier
            faces_rect, gray_img = faceDetection(test_img)

            # If in one image there are more than one faces then it is a problem for classifier 
            if (len(faces_rect)!=1):
                continue
            
            # Crop the face part in the image 
            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_img[y:y+w, x:x+h]

            # Append the facces and id related to the image 
            faces.append(roi_gray)
            faceID.append(int(id))

    
    return faces, faceID




def train_classifier(faces, faceID):

    # Binary descriptor
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the model 
    face_recognizer.train(faces, np.array(faceID))

    return face_recognizer



def draw_rect(test_img, face):
    (x, y, w, h) = face

    cv2.rectangle(test_img, (x,y),(x+w, y+h), (0, 255, 0), thickness = 5)


def put_text(test_img, text, x,y):
    cv2.putText(test_img, text, (x,y), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale= 1, color= (255, 0, 255), thickness=2)

