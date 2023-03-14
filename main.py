import logging
from logger import logger
import argparse

import cv2
import mediapipe as mp


parser = argparse.ArgumentParser()
parser.add_argument('--cam_num', type=int, default=0)
args = parser.parse_args()


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

HANDS_PARAMS = {
    "min_detection_confidence": 0.5, 
    "min_tracking_confidence": 0.5
}

FACE_MESH_PARAMS = {
    "max_num_faces" : 1,
    "refine_landmarks" : True,
    "min_detection_confidence" : 0.5,
    "min_tracking_confidence" : 0.5
}

if __name__ == "__main__":
    cap = cv2.VideoCapture(args.cam_num)
    hands = mp_hands.Hands(**HANDS_PARAMS)
    face_mesh = mp_face.FaceMesh(**FACE_MESH_PARAMS)
    with hands, face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            hands_results = hands.process(image)
            face_results = face_mesh.process(image)

            if hands_results.multi_hand_landmarks:
                 for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image, face_landmarks, mp_face.FACEMESH_CONTOURS)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('MediaPipe Hands', image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()