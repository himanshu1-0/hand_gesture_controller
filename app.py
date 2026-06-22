# app.py
import cv2
import pyautogui
import time
from gesture import get_landmarks, is_fist, is_open_hand, is_two_fingers_up

cap = cv2.VideoCapture(0)

last_action_time = time.time()
cooldown = 0.2 # seconds between actions

print("Hand Gesture Controller Started!")
print("✌️  Two fingers UP = Scroll Up")
print("✊  Close fingers  = Scroll Down")
print("✊  Fist = Volume Down")
print("🖐️  Open Hand = Volume Up")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    landmarks = get_landmarks(frame)

    now = time.time()
    action = ""

    if landmarks and (now - last_action_time) > cooldown:

        two_up = is_two_fingers_up(landmarks)
        fist = is_fist(landmarks)
        open_hand = is_open_hand(landmarks)

        if two_up:
            # Two fingers up = scroll UP
            pyautogui.scroll(40)
            action = "SCROLL UP"
            last_action_time = now

        elif fist:
            # Fist = Volume DOWN + Scroll DOWN
            pyautogui.press('volumedown')
            pyautogui.scroll(-40)
            action = "VOL DOWN / SCROLL DOWN"
            last_action_time = now

        elif open_hand:
            # Open hand = Volume UP
            pyautogui.press('volumeup')
            action = "VOL UP"
            last_action_time = now

    # Display gesture on screen
    color = (0, 255, 0) if "UP" in action else (0, 0, 255) if "DOWN" in action else (255, 255, 0)
    if action:
        cv2.putText(frame, action, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Instructions on screen
    cv2.putText(frame, "2 fingers=Scroll Up | Fist=Vol Down+Scroll Down | Open=Vol Up",
               (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("Hand Gesture Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()