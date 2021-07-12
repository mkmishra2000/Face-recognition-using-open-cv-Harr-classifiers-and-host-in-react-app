import face_recognition
import cv2
import sys
import os

known_faces_dir = "KnownFace"
unknown_faces_dir = "UnknownFace"
Tolerence = 0.3

frame_thickness =3

font_thickness =2

Model = "hog"

print("Loading known faces...")

known_faces = []
known_names = []

for name in os.listdir(known_faces_dir):
    for filename in os.listdir(f"{known_faces_dir}/{name}"):
        face_image = face_recognition.load_image_file(f"{known_faces_dir}/{name}/{filename}")
        encoding = face_recognition.face_encodings(face_image)


        known_faces.append(encoding)
        known_names.append(name)

    

print("Processing unknown faces")

for filename in os.listdir(unknown_faces_dir):
    print(filename)

    print("Doing check")

    image = face_recognition.load_image_file(f"{unknown_faces_dir}/{filename}")

    locations = face_recognition.face_locations(image, model=Model)

    encoding = face_recognition.face_encodings(image)[0]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_locations in zip(encoding, locations):

        # List of booleans 
        try:
            results = face_recognition.compare_faces(known_faces, face_encoding, Tolerence)
            print(results)
        except:
            print("Problem here...")

        
        match = None

        if True in results:
            match = known_names[results.index(True)]

            print(f"Match found: {match}")

            top_left = (face_locations[3], face_locations[0])
            bottom_right = (face_locations[1], face_locations[2])

            color = [0, 255, 0]

            cv2.rectangle(image, top_left , bottom_right, color, frame_thickness)

            top_left = (face_locations[3], face_locations[0])
            bottom_right = (face_locations[1], face_locations[2]+22)
            cv2.rectangle(image, top_left , bottom_right, color, frame_thickness)

            cv2.putText(image, match, (face_locations[3]+10, face_locations[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), font_thickness)

    
    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)










