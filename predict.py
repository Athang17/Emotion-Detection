import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2

def preprocess_image(img_path):
    #Loading the image
    img = image.load_img(img_path, target_size=(48,48),color_mode='grayscale')
    img_array = image.img_to_array(img)

    #Normalizing the image value
    img_array = img_array / 255.0

    #Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array,axis=0)

    return img_array

def predict_image(model,img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    emotion = np.argmax(predictions)
    return emotion

if __name__ == "__main__":
    #Load the model
    model = load_model('emotion_detection_model.h5')

    #Predict on a new image
    img_path = r"D:\NMIMS\Sem 5\IVP\Emotion-Detection\20240907_184429.jpg"
    emotion = predict_image(model,img_path)
    print(f"Predicted Emotion: {emotion}")