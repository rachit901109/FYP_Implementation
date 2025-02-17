import os
from keras._tf_keras.keras.preprocessing import image
import platform
import shutil
import numpy as np

UPLOAD_DIR = ""
MODEL_PATH = ""

# Paths
if platform.system()=="Windows":
    UPLOAD_DIR = r".\uploaded_files"
    MODEL_PATH = r".\model_weights\skin_dis_model.h5"
elif platform.system()=="Linux":
    UPLOAD_DIR = r"./uploaded_files"
    MODEL_PATH = r"./model_weights/skin_dis_model.h5"
print("Upload dir and model path:-")
print(UPLOAD_DIR, MODEL_PATH)

def save_file(file):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.name)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file, f)
    return file_path

def preprocess_image(img_path, target_size=(28, 28)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

