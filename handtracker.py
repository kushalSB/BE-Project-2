import cv2
import mediapipe as mp

#handtracking and drawing utility
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


#capturing from webcam
cap = cv2.VideoCapture(0)

hand_data=[] #data to store handlandmarks

#Main tracking loop
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

        # Draw hand landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                #saves data
                hand_data.append(hand_landmarks)
                print(hand_landmarks)
                print(type(hand_landmarks))
        # Display the resulting image
        cv2.imshow('Hand Tracking', image)

        # Exit the loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

file=open('data.txt','w')
for item in hand_data:
    file.write(str(item)+'\n')
file.close()
