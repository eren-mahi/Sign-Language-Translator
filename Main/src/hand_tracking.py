import cv2
import numpy as np
import tensorflow as tf
from keras.utils.image_utils import img_to_array

# Load the trained model
model = tf.keras.models.load_model('D:\\Sign Language Translator Using AI\\Main\\src\\models\\keras_model.h5') #Location of model
with open("D:\\Sign Language Translator Using AI\\Main\\src\\models\\labels.txt", "r") as f: #Location of the Label.txt which contains classification of alphabets from 1 to 26
    gesture_classes = [line.strip() for line in f.readlines()]

# Preprocess image for model
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Predict gesture
def recognize_gesture(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]

    if confidence > 0.7:
        return gesture_classes[predicted_class]
    else:
        return ""

# Main function that returns frame with prediction
def detect_hand_gesture(frame):
    # Define region of interest (ROI) — fixed box
    x, y, w, h = 100, 100, 300, 300
    roi = frame[y:y+h, x:x+w]

    # Predict gesture from ROI
    gesture = recognize_gesture(roi)

    # Draw box and label
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if gesture:
        cv2.putText(frame, f"Gesture: {gesture}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return frame, gesture
