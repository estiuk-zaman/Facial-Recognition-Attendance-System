import cv2
import numpy as np

# ১. নতুন ONNX মডেল লোড করা
model_path = "face_detection_yunet.onnx"

# ২. YuNet Face Detector তৈরি করা
face_detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(320, 320), # এটি পরে লাইভ ভিডিওর সাইজ অনুযায়ী আপডেট হবে
    score_threshold=0.6,   # ৬০% এর বেশি শিওর হলে তবেই মুখ মানবে
    nms_threshold=0.3,
    top_k=5000
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ফ্রেমের সাইজ (উচ্চতা, প্রস্থ) বের করা
    h, w = frame.shape[:2]
    
    # ৩. ডিটেক্টরকে বর্তমান ফ্রেমের আসল সাইজটা জানিয়ে দেওয়া
    face_detector.setInputSize((w, h))

    # ৪. মুখ খোঁজা (এটাই মেইন ম্যাজিক!)
    _, faces = face_detector.detect(frame)

    # ৫. মুখ পাওয়া গেলে বক্স আঁকা
    if faces is not None:
        for face in faces:
            # YuNet এর রেজাল্ট থেকে বক্সের x, y, width, height আলাদা করা
            box = face[0:4].astype(int)
            x, y, w_box, h_box = box
            
            # Confidence বা শিওরিটি স্কোর (লিস্টের একদম শেষের ভ্যালুটা)
            confidence = face[-1]

            # সবুজ বক্স আঁকা
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            
            # পার্সেন্টেজ লেখা
            text = f"{confidence*100:.1f}%"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('ONNX YuNet Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()