import cv2
import numpy as np
import os

# Directory setup for automated saving
user_name = input("Enter your name (e.g., Estiuk): ")
save_path = f"dataset/{user_name}"
os.makedirs(save_path, exist_ok=True)
print(f"Data will be saved in: {save_path}")

# Load ONNX model
model_path = "face_detection_yunet.onnx"
face_detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

cap = cv2.VideoCapture(0)
count = 0  # Image save counter

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
            
            # Safety boundary check to prevent array index out of bounds
            if x < 0 or y < 0 or x + w_box > w or y + h_box > h:
                continue

            # Array Slicing: Crop the facial region
            face_crop = frame[y:y+h_box, x:x+w_box]
            
            if face_crop.size != 0:
                # Convert to grayscale and normalize size for LBPH training
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (200, 200))

                # Save image to disk
                count += 1
                file_name = f"{save_path}/{count}.jpg"
                cv2.imwrite(file_name, gray_face)

                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(frame, f"Saved: {count}/100", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Day 3 - Data Collection', frame)
    
    # Auto-terminate after 100 samples
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Data Collection Successful! 100 images saved for {user_name}.")