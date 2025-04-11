import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import img_to_array

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Loading the pre-trained model
model = tf.keras.models.load_model('D:/Sign Language Translator Using AI/Main/src/models/gesture_model.h5')

# Gesture classes based on trained model
gesture_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def preprocess_image(image):
    image = cv2.resize(image, (200, 200))  # Resize to the input shape of the model
    image = img_to_array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def recognize_gesture(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)
    return gesture_classes[predicted_class[0]]

def detect_hand_gesture(frame):
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    landmark_frame = np.zeros_like(frame)  # Black background same size as frame

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            mp_drawing.draw_landmarks(
                landmark_frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Gesture recognition (unchanged)
            hand_image = np.zeros((200, 200, 3), dtype=np.uint8)
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * 200)
                y = int(landmark.y * 200)
                cv2.circle(hand_image, (x, y), 5, (255, 255, 255), -1)
            gesture = recognize_gesture(hand_image)
            cv2.putText(frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return frame, landmark_frame

