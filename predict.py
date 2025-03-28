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

def predict_image(model, img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)

    # Get the index of the highest predicted probability
    emotion_index = np.argmax(predictions)

    # Map the index to the corresponding emotion label
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion = emotion_labels[emotion_index]
    
    # Probabilities of all emotions
    emotion_probabilities = predictions[0]

    # Dictionary mapping emotions to their respective probabilities
    emotion_confidences = {emotion_labels[i]: round(emotion_probabilities[i] * 100, 2) for i in range(len(emotion_labels))}

    return emotion, emotion_confidences

if __name__ == "__main__":
    #Load the model
    model = load_model(r'D:\NMIMS\Sem 5\IVP\Emotion-Detection\emotion_detection_model.h5')

    #Predict on a new image
    img_path = r"C:\Users\Admin\Downloads\neutral_image.jpg"
    emotion, emotion_confidences = predict_image(model, img_path)  # Unpack both emotion and confidences

    print(f"Predicted Emotion: {emotion}")
    print("Emotion Confidences:")
    for emotion_label, confidence in emotion_confidences.items():
        print(f"{emotion_label}: {confidence}%")