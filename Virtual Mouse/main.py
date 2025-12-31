import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Safety
pyautogui.FAILSAFE = False

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe Hand Detector
hand_detector = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

drawing_utils = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Cursor smoothening
prev_x, prev_y = 0, 0
smoothening = 5

# Click control
last_left_click_time = 0
last_right_click_time = 0
click_delay = 0.5

# Scroll control
prev_scroll_y = 0
scroll_sensitivity = 15
smooth_scroll_factor = 3

# Finger positions
index_x, index_y = 0, 0
middle_x, middle_y = 0, 0
thumb_x, thumb_y = 0, 0
ring_x, ring_y = 0, 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                # Index finger - cursor
                if id == 8:
                    cv2.circle(frame, (x, y), 15, (255, 0, 0), 1)
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y

                    curr_x = prev_x + (index_x - prev_x) / smoothening
                    curr_y = prev_y + (index_y - prev_y) / smoothening
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

                # Thumb
                if id == 4:
                    cv2.circle(frame, (x, y), 15, (0, 0, 255), 1)
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y

                # Middle finger
                if id == 12:
                    cv2.circle(frame, (x, y), 15, (0, 255, 255), 1)
                    middle_x = screen_width / frame_width * x
                    middle_y = screen_height / frame_height * y

                # Ring finger
                if id == 16:
                    cv2.circle(frame, (x, y), 15, (255, 0, 255), 1)
                    ring_x = screen_width / frame_width * x
                    ring_y = screen_height / frame_height * y

            # LEFT CLICK (Thumb + Index)
            distance_thumb_index = math.hypot(index_x - thumb_x, index_y - thumb_y)
            if distance_thumb_index < 40:
                current_time = time.time()
                if current_time - last_left_click_time > click_delay:
                    pyautogui.click()
                    last_left_click_time = current_time
                cv2.putText(frame, "LEFT CLICK", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # SCROLL (Index + Middle close)
            distance_index_middle = math.hypot(index_x - middle_x, index_y - middle_y)
            if distance_index_middle < 50:
                if prev_scroll_y != 0:
                    scroll_delta = (index_y - prev_scroll_y) / smooth_scroll_factor
                    pyautogui.scroll(int(-scroll_delta * scroll_sensitivity))
                prev_scroll_y += (index_y - prev_scroll_y) / smooth_scroll_factor
                cv2.putText(frame, "SCROLL", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                prev_scroll_y = index_y

            # RIGHT CLICK (Thumb + Ring)
            distance_thumb_ring = math.hypot(thumb_x - ring_x, thumb_y - ring_y)
            if distance_thumb_ring < 40:
                current_time = time.time()
                if current_time - last_right_click_time > click_delay:
                    pyautogui.click(button='right')
                    last_right_click_time = current_time
                cv2.putText(frame, "RIGHT CLICK", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
