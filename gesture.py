# gesture.py
import mediapipe as mp
import cv2
import os
import urllib.request

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        model_path
    )
    print("Model downloaded!")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)


def get_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    landmarks = []
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for lm in hand:
                landmarks.append((lm.x, lm.y))
    return landmarks


def count_fingers_up(landmarks):
    if not landmarks:
        return 0
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    count = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip][1] < landmarks[pip][1]:
            count += 1
    return count


def is_fist(landmarks):
    return count_fingers_up(landmarks) == 0


def is_open_hand(landmarks):
    return count_fingers_up(landmarks) == 4


def is_two_fingers_up(landmarks):
    if not landmarks:
        return False
    index_up = landmarks[8][1] < landmarks[6][1]
    middle_up = landmarks[12][1] < landmarks[10][1]
    ring_down = landmarks[16][1] > landmarks[14][1]
    pinky_down = landmarks[20][1] > landmarks[18][1]
    return index_up and middle_up and ring_down and pinky_down