import cv2
import numpy as np

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
            
            
            confidence = face[-1]

            
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            
            
            text = f"{confidence*100:.1f}%"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('ONNX YuNet Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()