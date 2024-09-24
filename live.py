import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained emotion detection model
model = load_model(r'D:\NMIMS\Sem 5\IVP\Emotion-Detection\emotion_detection_model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess the image for emotion detection
def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)  # Add channel dimension for grayscale
    return face

# Function to detect emotion in a face
def detect_emotion(face):
    preprocessed_face = preprocess_face(face)
    predictions = model.predict(preprocessed_face)
    emotion_index = np.argmax(predictions)
    emotion_confidences = predictions[0]
    emotion = emotion_labels[emotion_index]
    return emotion, emotion_confidences

# Start the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a named window for display
cv2.namedWindow('Emotion Detection')

while True:  # Infinite loop, will stop when 'q' is pressed
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each face
    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]

        # Predict emotion for the detected face
        emotion, _ = detect_emotion(face)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Put the emotion label above the rectangle
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
