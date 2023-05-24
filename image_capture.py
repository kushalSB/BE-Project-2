import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize hand tracking
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Specify the symbols or signs and their corresponding image names
symbol_mapping = {
    'thumbs_up': 'thumbs_up.jpg',
    'peace': 'peace.jpg',
    # Add more symbols and their image names here

    #Add image from kaggle asti hereko letter wala with their jpgs in the same folder
    # ani press s to capture and terminal ma gara name hala
    #this step is for data collection
}

# Specify the desired landmark indices to store for each symbol
desired_landmark_indices = {
    'thumbs_up': [4, 8, 12, 16, 20],
    'peace': [4, 8, 12, 16, 20],
    # Add more symbols and their desired landmark indices here
}

# Function to save specific landmarks for a given symbol
def save_landmarks(symbol, landmarks):
    image_name = symbol_mapping.get(symbol)
    if image_name is None:
        print(f"Symbol '{symbol}' does not have a corresponding image name.")
        return
    landmark_indices = desired_landmark_indices.get(symbol)
    if landmark_indices is None:
        print(f"Symbol '{symbol}' does not have desired landmark indices specified.")
        return
    with open(image_name, 'a') as file:
        file.write(symbol + ': ')
        for index in landmark_indices:
            landmark = landmarks.landmark[index]
            file.write(f"{landmark.x},{landmark.y},{landmark.z} ")
        file.write('\n')
    print(f"Landmarks for symbol '{symbol}' saved successfully.")

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read video source.")
        break

    # Convert image to RGB and process it
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if any key is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Prompt user to enter the symbol name and save the landmarks
                symbol = input("Enter the symbol name: ")
                save_landmarks(symbol, hand_landmarks)

    # Show the image
    cv2.imshow('Hand Landmark Capture', image)

# Release resources
cap.release()
cv2.destroyAllWindows()
