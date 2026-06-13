import cv2
import face_recognition
import csv
from datetime import datetime

# 1. Load a known face and get its encoding
image_of_person = face_recognition.load_image_file("person.jpg")
person_encoding = face_recognition.face_encodings(image_of_person)[0]

known_face_encodings = [person_encoding]
known_face_names = ["John Doe"]

# 2. Open a CSV to log attendance
csv_file = open('attendance.csv', 'a+', newline='')
writer = csv.writer(csv_file)

# 3. Start Camera
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces and encode
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            
            # Log attendance if match found
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([name, dt_string])
            print(f"Attendance recorded for: {name}")

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
csv_file.close()
