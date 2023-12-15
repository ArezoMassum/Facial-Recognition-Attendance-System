import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture=cv2.VideoCapture(0)

arezo_image= face_recognition.load_image_file("students/Arezo.jpg")
arezo_data=face_recognition.face_encodings(arezo_image)[0]

ece_image= face_recognition.load_image_file("students/Ece.jpg")
ece_data=face_recognition.face_encodings(ece_image)[0]

aisha_iamge= face_recognition.load_image_file("students/Aisha.jpg")
aisha_data=face_recognition.face_encodings(aisha_iamge)[0]

melike_image= face_recognition.load_image_file("students/Melike.jpg")
melieke_data=face_recognition.face_encodings(melike_image)[0]

known_face_data=[
arezo_data,
ece_data,
aisha_data,
melieke_data]

known_face_names=[
    "Arezo Massum",
    "Ecenaz Güngör",
    "Aisha Hekmat",
    "Melike Yaman"]

students=known_face_names.copy()
face_locations=[]
face_data= []
face_names=[]
s= True

current=datetime.now()
current_date=current.strftime("%Y-%m-%d")


f=open(current_date+'.csv', 'w+', newline='')
Inwriter=csv.writer(f)

while True:

    ret, frame=video_capture.read()

    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    if s:
        face_location=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(known_face_data, face_encoding)
            name=""
            face_distance=face_recognition.face_distance(known_face_data, face_encoding)
            best_match_index=np.argmin(face_location)
            if matches[best_match_index]:
                name=known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time=current.strftime("%H-%M-%S")
                    Inwriter.writerow([name,current_time])
    cv2.imshow("attendence system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()