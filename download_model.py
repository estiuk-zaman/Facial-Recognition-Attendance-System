import urllib.request
import os

# haarcascades folder toiri kora
os.makedirs('haarcascades', exist_ok=True)

# XML model download kora
url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
urllib.request.urlretrieve(url, "haarcascades/haarcascade_frontalface_default.xml")
print("Model Download Completed!")