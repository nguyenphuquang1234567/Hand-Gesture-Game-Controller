import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    prev_action = ""
    last_time = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)

        image_height, image_width, _ = image.shape
        action = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x = wrist.x * image_width
                wrist_y = wrist.y * image_height

                # Xác định vùng tay để điều khiển
                if wrist_x < image_width * 0.3:
                    action = "left"
                elif wrist_x > image_width * 0.7:
                    action = "right"
                elif wrist_y < image_height * 0.4:
                    action = "up"
                elif wrist_y > image_height * 0.7:
                    action = "down"

        # Ngăn spam phím quá nhanh
        current_time = time.time()
        if action and (action != prev_action or current_time - last_time > 1):
            pyautogui.press(action)
            prev_action = action
            last_time = current_time

        # Hiển thị hành động đang thực hiện
        cv2.putText(image, f"Action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Subway Surfer Hand Control", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()