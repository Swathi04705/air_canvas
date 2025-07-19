import cv2
import numpy as np
import mediapipe as mp

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Drawing parameters
draw_color = (0, 0, 255)  # Start with red
brush_thickness = 15
eraser_thickness = 50
xp, yp = 0, 0

# Canvas
canvas = np.zeros((720, 1280, 3), np.uint8)

# Mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Color palette positions
palette_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)]
palette_pos = [(40 + i * 100, 40) for i in range(len(palette_colors))]
selected_color = draw_color

# Utility to detect which palette color is selected
def get_palette_selection(x, y):
    for idx, (px, py) in enumerate(palette_pos):
        if px - 30 < x < px + 30 and py - 30 < y < py + 30:
            return palette_colors[idx]
    return None

def fingers_up(lmList):
    fingers = []
    if lmList[4][0] < lmList[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)
    for tip in [8, 12, 16, 20]:
        if lmList[tip][1] < lmList[tip - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
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
        x1, y1 = lmList[8]  # Index fingertip

        # Check for palette selection
        if fingers == [0, 1, 0, 0, 0]:
            color = get_palette_selection(x1, y1)
            if color:
                draw_color = color

            # Draw only after moving out of palette zone
            if not get_palette_selection(x1, y1):
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                thickness = eraser_thickness if draw_color == (0, 0, 0) else brush_thickness
                cv2.line(frame, (xp, yp), (x1, y1), draw_color, thickness)
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)
                xp, yp = x1, y1
        else:
            xp, yp = 0, 0

        # Clear canvas gesture
        if fingers == [1, 1, 1, 1, 1]:
            canvas = np.zeros_like(frame)

    # Overlay palette
    for i, (px, py) in enumerate(palette_pos):
        cv2.circle(frame, (px, py), 30, palette_colors[i], -1)
        if palette_colors[i] == draw_color:
            cv2.rectangle(frame, (px - 35, py - 35), (px + 35, py + 35), (255, 255, 255), 3)

    # Overlay canvas
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.add(frame, canvas)

    cv2.imshow("Air Canvas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("saved_canvas.png", canvas)
        print("Canvas saved as saved_canvas.png")



cap.release()
cv2.destroyAllWindows()
