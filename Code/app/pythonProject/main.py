import face_recognition
import cv2
import numpy as np
import csv                                                       
import os
from datetime import datetime
'''Importing required files:
cv = capturing video/
numpy = advanced numbers library/
csv = processsing image and video/
datetime = undertaking current system date and time/'''

'''Capture the video that need to be processed for attendance/'''
video_capture = cv2.VideoCapture(0)

'''Face recognize the image stored/'''
ajay_image = face_recognition.load_image_file("photos/ajay.jpg")
ajay_encoding = face_recognition.face_encodings(ajay_image)[0]

praveen_image = face_recognition.load_image_file("photos/praveen.jpg")
praveen_encoding = face_recognition.face_encodings(praveen_image)[0]

arul_image = face_recognition.load_image_file("photos/arul.jpg")
arul_encoding = face_recognition.face_encodings(arul_image)[0]

rohith_image = face_recognition.load_image_file("photos/rohith.jpg")
rohith_encoding = face_recognition.face_encodings(rohith_image)[0]

'''leena_mam_image = face_recognition.load_image_file("photos/leena_mam.jpg")
leena_mam_encoding = face_recognition.face_encodings(leena_mam_image)[0]

mydhili_mam_image = face_recognition.load_image_file("photos/mydhili_mam.jpg")
mydhili_mam_encoding = face_recognition.face_encodings(mydhili_mam_image)[0]'''

known_face_encoding = [
ajay_encoding,
praveen_encoding,
arul_encoding,
rohith_encoding
]

known_faces_names = [
"ajay",
"praveen",
"arul",
"rohith"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings (rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2

                cv2.putText(frame,name+' Present',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

