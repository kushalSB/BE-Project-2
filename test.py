import cv2
import mediapipe as mp

#handtracking and drawing utility
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


#capturing from webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=4,
        min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        # Read the frame from the video source
        success, image = cap.read()
        if not success:
            print("Failed to read video source")
            break

        # Convert the image to RGB and process it
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        print(results)
        print(type(results))