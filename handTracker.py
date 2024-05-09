import cv2 
import mediapipe as mp
import time

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW) #// if you have second camera you can set first parameter as 1

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

if not (cap.isOpened()):
    print("Could not open video device")
while True: 
    ret,frame= cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm, in enumerate(handLms.landmark):
                # print(id, lm)
                # height, width, channels
                h, w, c = frame.shape
                # position of center
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 8:
                  cv2.circle(frame, (cx, cy), 15, (255, 255, 255), cv2.FILLED)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)


    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 255, 255), 3)

    cv2.imshow("Live",frame)
    cv2.waitKey(1)
cv2.destroyAllWindows()
