import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, _ = img.shape
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                myHand = {}
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    lmList.append([px, py])
                myHand["lmList"] = lmList
                allHands.append(myHand)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return allHands

    def fingersUp(self, hand):
        fingers = []
        lmList = hand["lmList"]
        if not lmList:
            return []

        # Thumb
        if lmList[self.tipIds[0]][0] < lmList[self.tipIds[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers (index to pinky)
        for id in range(1, 5):
            if lmList[self.tipIds[id]][1] < lmList[self.tipIds[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
