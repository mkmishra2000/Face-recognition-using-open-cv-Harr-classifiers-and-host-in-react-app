

from flask import Flask, render_template, send_from_directory, request, jsonify, send_file, make_response
import cv2
from PIL import Image
import io
import os
# import ImagefaceDetection as ifd
import FaceTest as fr
import numpy as np
import cv2
import base64
from matplotlib import cm
from io import BytesIO

app = Flask(__name__)



def detectFace(test_img):

    face_detected, grayImage = fr.faceDetection(test_img)

    print("Face detected : ", face_detected)

    faceReconizer = cv2.face.LBPHFaceRecognizer_create()

    faceReconizer.read("/home/manas/Documents/ImageProcessing/project/faceRec/trainingData.yml")

    name = {0:"MANAS",
            1: "VIJAY",
            2:"Sachine"}

    returnImg = test_img

    print("Everthing is ok...")
    for face in face_detected:
        (x,y,w,h) = face

        print("I m in the loop")
        roi_gray = grayImage[y:y+h, x:x+h]

        label, confidence = faceReconizer.predict(roi_gray)

        print("Confidence : ", confidence)
        print("Probability of correct :", 100-confidence)
        print("Label :", label)
        print("Predicted name : ", name[label])

        fr.draw_rect(returnImg, face)

        predicted_name=name[label]


        if confidence> 70:
            continue

        fr.put_text(returnImg, predicted_name , x, y)
        

    return returnImg


@app.route('/face-detection',methods = ["POST"])
def imageInput():

    print("Input from the react ") 

    data = request.files.get('image')

    f = data.stream.read()
    bin_data = io.BytesIO(f)

    file_bytes = np.asarray(bytearray(bin_data.read()), dtype = np.uint8)

    img2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if(data):
        print("working")
        returnImg = detectFace(img2)
        resized_img = cv2.resize(returnImg, (1000, 700))

        cv2.imshow("Face detection", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # b = returnImg.tolist()

        # fook = io.BytesIO(base64.b64encode(returnImg))
        # img = Image.fromarray(returnImg, 'RGB')
        
        # imgByteArr = io.BytesIO(returnImg)
        # imgByteArr = imgByteArr.getvalue()
        # imgByteArr = base64.encodebytes(imgByteArr).decode('ascii')
        # something = False
        # cv2.imwrite(something,returnImg)
        # return jsonify({"image": something })
        # img = returnImg.tostring()
        # img = Image.fromarray(returnImg)
        cv2.imwrite("./frontend/dip--project/src/thenewimg.jpeg", returnImg)

        response = make_response(send_file("/home/manas/Documents/ImageProcessing/project/frontend/dip--project/src/thenewimg.jpeg",mimetype='image/png'))
        response.headers['Content-Transfer-Encoding']='base64'
        return response
    else:
        return jsonify({"error": 'invalid data', "code": 500})
    
    # return jsonify({"image": returnImg})



if __name__ == "__main__":
    app.run(host="localhost",port= 6000, debug=True)

