import cv2
import numpy as np
import mediapipe as mp

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Drawing parameters
draw_color = (0, 0, 255)  # Default: red
brush_thickness = 15
eraser_thickness = 50
brush_mode = "normal"  # "normal", "dotted", "dashed"
xp, yp = 0, 0

# Canvas to draw on
canvas = np.zeros((720, 1280, 3), np.uint8)

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Track brush mode toggle state
last_toggle = False

def fingers_up(lmList):
    fingers = []
    # Thumb
    fingers.append(1 if lmList[4][0] < lmList[3][0] else 0)
    # Index to pinky
    for tip in [8, 12, 16, 20]:
        fingers.append(1 if lmList[tip][1] < lmList[tip - 2][1] else 0)
    return fingers

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    lmList = []
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

    if lmList:
        fingers = fingers_up(lmList)

        # Clear canvas with 5 fingers
        if fingers == [1, 1, 1, 1, 1]:
            canvas = np.zeros_like(canvas)
            cv2.putText(frame, 'Canvas Cleared', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

        # Color selection gestures
        elif fingers == [0, 1, 1, 0, 0]:
            draw_color = (255, 0, 0)  # Blue
        elif fingers == [0, 1, 1, 1, 0]:
            draw_color = (0, 255, 0)  # Green
        elif fingers == [0, 1, 1, 1, 1]:
            draw_color = (0, 0, 255)  # Red

        # Eraser mode (thumb + index)
        elif fingers == [1, 1, 0, 0, 0]:
            draw_color = (0, 0, 0)

        # Toggle brush mode (middle + ring finger up)
        elif fingers == [0, 0, 1, 1, 0]:
            if not last_toggle:
                if brush_mode == "normal":
                    brush_mode = "dotted"
                elif brush_mode == "dotted":
                    brush_mode = "dashed"
                else:
                    brush_mode = "normal"
                last_toggle = True
        else:
            last_toggle = False

        # Drawing (index finger only)
        if fingers == [0, 1, 0, 0, 0]:
            x1, y1 = lmList[8]
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            thickness = eraser_thickness if draw_color == (0, 0, 0) else brush_thickness

            if draw_color == (0, 0, 0):  # Eraser mode
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)
            else:
                if brush_mode == "dotted":
                    cv2.circle(canvas, (x1, y1), thickness // 2, draw_color, -1)
                elif brush_mode == "dashed":
                    dist = int(np.hypot(x1 - xp, y1 - yp))
                    for i in range(0, dist, 30):
                        t = i / dist
                        xi = int(xp + t * (x1 - xp))
                        yi = int(yp + t * (y1 - yp))
                        cv2.circle(canvas, (xi, yi), thickness // 2, draw_color, -1)
                else:
                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)

            xp, yp = x1, y1
        else:
            xp, yp = 0, 0

        # Brush mode display
        cv2.putText(frame, f'Brush: {brush_mode}', (1050, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

        # Show current selected color
        cv2.rectangle(frame, (0, 0), (100, 100), draw_color, -1)

    # Merge canvas onto frame
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.add(frame, canvas)

    cv2.imshow("Air Canvas", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        cv2.imwrite("drawing_output.png", canvas)
        print("Canvas saved as drawing_output.png")

cap.release()
cv2.destroyAllWindows()
