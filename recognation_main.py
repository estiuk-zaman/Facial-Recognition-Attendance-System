import cv2
import numpy as np

# ১. মডেল এবং ডেটা লোড করা
# YuNet (মুখ খোঁজার জন্য)
face_detector = cv2.FaceDetectorYN.create("face_detection_yunet.onnx", "", (320, 320), 0.6, 0.3, 5000)

# LBPH (মুখ চেনার জন্য)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml') # আমাদের ট্রেইন করা ব্রেইন লোড করলাম

# নামগুলো (ID থেকে Name) লোড করলাম
names = np.load('names.npy', allow_pickle=True).item() 

cap = cv2.VideoCapture(0)

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
            
            # সেফটি চেক
            if x < 0 or y < 0 or x + w_box > w or y + h_box > h: 
                continue
            
            face_crop = frame[y:y+h_box, x:x+w_box]
            if face_crop.size == 0: 
                continue
            
            # আগের মতোই ফেস প্রসেস করা (ট্রেনিংয়ের সময় যেমন ২০০x২০০ ছিল, ঠিক তেমনটাই দিতে হবে)
            gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (200, 200))
            
            # ২. আসল ম্যাজিক: মডেলকে জিজ্ঞেস করা "এই মুখটা কার?"
            id_, distance = recognizer.predict(gray_face)
            
            # ৩. লজিক: দূরত্ব (Distance) যত কম, ম্যাচিং তত ভালো!
            if distance < 80:  
                name = names.get(id_, "Unknown")
                color = (0, 255, 0) # সবুজ বক্স (চিনতে পেরেছে)
            else:
                name = "Unknown"
                color = (0, 0, 255) # লাল বক্স (চিনতে পারেনি)
                
            # স্ক্রিনে বক্স এবং নাম দেখানো
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, f"{name} ({int(distance)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Day 4 - Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()