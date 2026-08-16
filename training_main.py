import cv2
import numpy as np
import os


dataset_path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []
names = {}
current_id = 0

print("Training started... Please wait.")


for user_name in os.listdir(dataset_path):
    user_folder = os.path.join(dataset_path, user_name)
    
    if os.path.isdir(user_folder):
        names[current_id] = user_name 
        
        
        for image_name in os.listdir(user_folder):
            image_path = os.path.join(user_folder, image_name)
            
            
            face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if face_image is None:
                continue
            
            faces.append(face_image)
            ids.append(current_id)
            
        current_id += 1


print(f"Training on {len(faces)} images...")
recognizer.train(faces, np.array(ids))


recognizer.write('trainer.yml')
np.save('names.npy', names) 

print("✅ Training Successful! Model saved as 'trainer.yml'")
print("Names mapped:", names)