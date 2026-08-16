import cv2
import numpy as np
import os
import csv
from datetime import datetime

face_detector = cv2.FaceDetectorYN.create("face_detection_yunet.onnx", "", (320, 320), 0.6, 0.3, 5000)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

names = np.load('names.npy', allow_pickle=True).item() 

cap = cv2.VideoCapture(0)

# --- DAY 6: মেমোরি বক্স তৈরি করা ---
marked_names = set() 
print("System is Ready. Looking for faces...")
if not os.path.isfile('Attendance.csv'):
    with open('Attendance.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time", "Date"])
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame)

    if faces is not None:
        for face in faces:
            box = face[0:4].astype(int)
            x, y, w_box, h_box = box
            
            if x < 0 or y < 0 or x + w_box > w or y + h_box > h: 
                continue
            
            face_crop = frame[y:y+h_box, x:x+w_box]
            if face_crop.size == 0: 
                continue
            
            gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (200, 200))
            
            id_, distance = recognizer.predict(gray_face)
            
            if distance < 80:  
                name = names.get(id_, "Unknown")
                color = (0, 255, 0) 
                
                # --- DAY 6: DUPLICATION FIX LOGIC ---
                # যদি নামটা "Unknown" না হয় এবং মেমোরি বক্সে না থাকে
                if name != "Unknown" and name not in marked_names:
                    now = datetime.now()
                    time_str = now.strftime("%H:%M:%S")
                    date_str = now.strftime("%d-%m-%Y")
                    
                    with open('Attendance.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([name, time_str, date_str])
                    
                    # নামটা মেমোরি বক্সে সেভ করে রাখলাম যাতে আজ আর এন্ট্রি না হয়
                    marked_names.add(name)
                    print(f"Attendance recorded for {name} at {time_str}")
                # ------------------------------------

            else:
                name = "Unknown"
                color = (0, 0, 255) 
                
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, f"{name} ({int(distance)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Day 6 - Smart Attendance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()