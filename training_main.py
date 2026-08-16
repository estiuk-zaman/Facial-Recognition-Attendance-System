import cv2
import numpy as np
import os

# ১. ডেটাসেট ফোল্ডারের পাথ
dataset_path = 'dataset'

# ২. LBPH Recognizer তৈরি করা (এটাই আমাদের ব্রেইন)
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []
names = {} # ID এর সাথে নাম মনে রাখার জন্য (যেমন: 0 = Estiuk)
current_id = 0

print("Training started... Please wait.")

# ৩. dataset ফোল্ডারের ভেতরের ছবিগুলো রিড করা
for user_name in os.listdir(dataset_path):
    user_folder = os.path.join(dataset_path, user_name)
    
    if os.path.isdir(user_folder):
        names[current_id] = user_name # নামটা ডিকশনারিতে সেভ করে রাখলাম
        
        # ইউজারের ১০০টি ছবি একে একে পড়া
        for image_name in os.listdir(user_folder):
            image_path = os.path.join(user_folder, image_name)
            
            # ছবিটা গ্রে-স্কেল মোডে ওপেন করা
            face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if face_image is None:
                continue
            
            faces.append(face_image)
            ids.append(current_id)
            
        current_id += 1

# ৪. আসল ম্যাজিক: মডেলকে ট্রেইন করা
print(f"Training on {len(faces)} images...")
recognizer.train(faces, np.array(ids))

# ৫. ট্রেইন করা ব্রেইনটাকে হার্ডড্রাইভে সেভ করা
recognizer.write('trainer.yml')
np.save('names.npy', names) # নামগুলোও সেভ করে রাখলাম কালকের জন্য

print("✅ Training Successful! Model saved as 'trainer.yml'")
print("Names mapped:", names)