#Doesn't work yet


import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Load the pre-trained CNN model
model = keras.models.load_model('hand_detection_model.h5')

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read video source")
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert the hand landmarks to numpy array
                landmarks = np.array([(lmk.x, lmk.y, lmk.z) for lmk in hand_landmarks.landmark])
                
                # Preprocess the landmarks for input to the CNN model
                # Modify the preprocessing steps based on your requirements
                processed_landmarks = (landmarks - landmarks.mean(axis=0)) / landmarks.std(axis=0)
                processed_landmarks = np.expand_dims(processed_landmarks, axis=0)

                # Perform hand detection using the CNN model
                prediction = model.predict(processed_landmarks)
                if prediction > 0.5:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
