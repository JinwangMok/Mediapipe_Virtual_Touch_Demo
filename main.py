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

TARGET_FACE_LANDMARKS = {
    "left_eye_tip" :  263,
    "right_eye_tip" : 33
}

TARGET_HAND_LANDMARKS = {
    "index_finger_tip": 8
}

if __name__ == "__main__":
    cap = cv2.VideoCapture(args.cam_num)
    hands = mp_hands.Hands(**HANDS_PARAMS)
    face_mesh = mp_face.FaceMesh(**FACE_MESH_PARAMS)
    with hands, face_mesh:
        while cap.isOpened():
            success, camera_frame = cap.read()
            if not success:
                break

            camera_frame = cv2.cvtColor(cv2.flip(camera_frame, 1), cv2.COLOR_BGR2RGB)

            hands_results = hands.process(camera_frame)
            face_results = face_mesh.process(camera_frame)

            landmark_data = {}
            if hands_results.multi_hand_landmarks:
                 for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(camera_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for landmark_name, landmark_num in TARGET_HAND_LANDMARKS.items():
                        x = hand_landmarks.landmark[landmark_num].x
                        y = hand_landmarks.landmark[landmark_num].y
                        z = hand_landmarks.landmark[landmark_num].z + 1.
                        landmark_data[landmark_name] = (x, y, z)
            
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(camera_frame, face_landmarks, mp_face.FACEMESH_CONTOURS)
                    for landmark_name, landmark_num in TARGET_FACE_LANDMARKS.items():
                        x = face_landmarks.landmark[landmark_num].x
                        y = face_landmarks.landmark[landmark_num].y
                        z = face_landmarks.landmark[landmark_num].z + 1.
                        landmark_data[landmark_name] = (x, y, z)

            camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2BGR)
            if len(landmark_data) > 0:
                for i, (landmark_name, coordinate) in enumerate(landmark_data.items()):
                    cv2.putText(camera_frame, f"{landmark_name}_x: {coordinate[0]:.3f}", (10, (90*(i+1))+0), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(camera_frame, f"{landmark_name}_y: {coordinate[1]:.3f}", (10, (90*(i+1))+30), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(camera_frame, f"{landmark_name}_z: {coordinate[2]:.3f}", (10, (90*(i+1))+60), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('MediaPipe Hands', camera_frame)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()