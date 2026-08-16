import urllib.request

print("Downloading YuNet ONNX Model (approx 1.7 MB)...")
url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
urllib.request.urlretrieve(url, "face_detection_yunet.onnx")
print("Download Complete! You are ready to use the ONNX model.")