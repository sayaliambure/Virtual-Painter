import cv2
import numpy as np
import time, os
import handTrackerMod as htm

brushThickness = 15
eraserThickness = 50

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []

for imgPath in myList:
  image = cv2.imread(f'{folderPath}/{imgPath}')
  overlayList.append(image)

# print(len(overlayList))
header = overlayList[0]
drawColor = (0,0,255)
# the color scheme is BGR

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW) #// if you have second camera you can set first parameter as 1

cap.set(3, 1280)  # set cam size
cap.set(4, 720)
# cap.set(3, 800)  # set cam size
# cap.set(4, 620)

detector = htm.handDetector(detectionConf=0.85)
xprev, yprev = 0, 0 

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

if not (cap.isOpened()):
    print("Could not open video device")
while True: 
    # 1. import img
    ret,frame= cap.read()
    frame = cv2.flip(frame, 1)

    # 2. find hand landmarks
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList)!=0:
      # print(lmList)

      # tip of index and middle finger
      x1, y1 = lmList[8][1:]    # index finger
      x2, y2 = lmList[12][1:]   # middle finger
    

      # 3. check which fingers are up
      fingers = detector.fingersUp()
      # print(fingers)

      # 4. if selection mode- 2 fingers up
      if fingers[1] and fingers[2]:
        xprev, yprev = 0,0

        cv2.rectangle(frame, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)
        # print('selection mode')
        # print(x1)

        # if we go into header location
        if y1 < 75:
          if 65 < x1 < 125:
             header = overlayList[0]
            #  red
             drawColor = (0,0,255) #BGR   

          elif 150 < x1 < 215:
             header = overlayList[1]
            #  blue
             drawColor = (255, 195, 0)

          elif 240 < x1 < 305:
             header = overlayList[2]
            #  green
             drawColor = (139, 195, 74)

          elif 330 < x1 < 390:
             header = overlayList[3]
            #  yellow
             drawColor = (88, 238, 255)

          elif 420 < x1 < 480:
             header = overlayList[4]
             drawColor = (0,0,0)


          elif 545 < x1 < 595:
            brushThickness = 5
            # cv2.putText(frame, str(int(brushThickness)), (200,200), cv2.FONT_HERSHEY_PLAIN, 3,
            #     (255, 255, 255), 3)
            cv2.putText(frame, f"Brush Thickness: {brushThickness}",
                (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

          elif 620 < x1 < 665:
            brushThickness = 15
            cv2.putText(frame, f"Brush Thickness: {brushThickness}",
                (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
          elif 695 < x1 < 735:
            brushThickness = 25
            cv2.putText(frame, f"Brush Thickness: {brushThickness}",
                (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
             
      
      # 5. drawing mode- 1 finger up
      if fingers[1] and fingers[2]==False:
        cv2.circle(frame, (x1, y1), 15, drawColor, cv2.FILLED)
        #  print('drawing mode')

        # initially, to avoid a unnecesary line we make it a point
        if xprev == 0 and yprev == 0:
           xprev, yprev = x1, y1

        if drawColor == (0,0,0):
          cv2.line(frame, (xprev, yprev), (x1, y1), drawColor, eraserThickness)
          cv2.line(imgCanvas, (xprev, yprev), (x1, y1), drawColor, eraserThickness)

        else:
          cv2.line(frame, (xprev, yprev), (x1, y1), drawColor, brushThickness)
          cv2.line(imgCanvas, (xprev, yprev), (x1, y1), drawColor, brushThickness)

        xprev, yprev = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)  # convert to gray img
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)   # into binary image and inversing it
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)   #converting again to colored img as we need to add it to coloured img (gray cannot be added to coloured img)
    frame = cv2.bitwise_and(frame, imgInverse)
    frame = cv2.bitwise_or(frame, imgCanvas)

    # setting header img
    frame[0:70, 0:800] = header

    # add 2 images together ie the frame and the black screen on which we draw
    # frame = cv2.addWeighted(frame, 0.5, imgCanvas, 0.5, 0)


    cv2.imshow("Live",frame)
    # cv2.imshow("Canvas",imgCanvas)
    # cv2.imshow("Canvas",imgInverse)

    cv2.waitKey(1)
    cv2.resizeWindow("Live", 800, 550)
cv2.destroyAllWindows()
