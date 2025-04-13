import cv2
import mediapipe as mp
import numpy as np

class HandTracking:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.cap = cv2.VideoCapture(0)
        self.results = None

        # Drawing specs for lines and circles
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)

    def is_finger_folded(self, landmarks, finger_tip_idx, finger_mcp_idx):
        return landmarks[finger_tip_idx].y > landmarks[finger_mcp_idx].y

    def recognize_gesture(self, landmarks):
        if (self.is_finger_folded(landmarks, 8, 5) and
            self.is_finger_folded(landmarks, 12, 9) and
            self.is_finger_folded(landmarks, 16, 13) and
            self.is_finger_folded(landmarks, 20, 17)):
            return "A"

        if (not self.is_finger_folded(landmarks, 8, 5) and
            not self.is_finger_folded(landmarks, 12, 9) and
            not self.is_finger_folded(landmarks, 16, 13) and
            not self.is_finger_folded(landmarks, 20, 17)):
            return "B"

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinky_tip = landmarks[20]

        thumb_pinky_dist = np.sqrt((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2)
        thumb_index_dist = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

        if thumb_pinky_dist > thumb_index_dist:
            return "C"

        return None

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return frame, None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                gesture = self.recognize_gesture(landmarks)
                return frame, gesture

        return frame, None

    def get_landmark_only_frame(self):
        blank = np.zeros((200, 200, 3), dtype=np.uint8)

        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    blank,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.drawing_spec,
                    self.drawing_spec
                )
        return blank


    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()
