import cv2 
import mediapipe as mp
import time

class handDetector():
  def __init__(self, mode=False, maxHands=3, complexity=1, detectionConf=0.5, trackCon=0.5):
    self.mode = mode
    self.maxhands = maxHands
    self.complexity = complexity
    self.detectionConf = detectionConf
    self.trackCon = trackCon

    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands(self.mode, self.maxhands, self.complexity, self.detectionConf, self.trackCon)
    self.mpDraw = mp.solutions.drawing_utils

    self.tipIds = [4, 8, 12, 16, 20]

  def findHands(self, frame, draw=True):

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)

    if self.results.multi_hand_landmarks:
        # gets landmark for all the hands
        for handLms in self.results.multi_hand_landmarks:
            if draw:
              self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

    return frame
  

  def findPosition(self, frame, handNo=0, draw=True):

    self.lmList = []
    if self.results.multi_hand_landmarks:  
      # get landmark of hand which is specified
      myHand = self.results.multi_hand_landmarks[handNo]

      for id, lm, in enumerate(myHand.landmark):
          # height, width, channels
          h, w, c = frame.shape
          # position of center
          cx, cy = int(lm.x*w), int(lm.y*h)
          self.lmList.append([id, cx, cy])
          if draw:
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), cv2.FILLED)

    return self.lmList
  

  def fingersUp(self):
    fingers = []

    #  thumb
    if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
     fingers.append(1)
    else:
     fingers.append(0)

    # 4 fingers
    for id in range(1, 5):
      if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
        fingers.append(1)
      else:
        fingers.append(0)
    
    return fingers


def main():

  prevTime = 0
  currTime = 0

  cap=cv2.VideoCapture(0,cv2.CAP_DSHOW) #// if you have second camera you can set first parameter as 1    
  detector = handDetector()
  
  if not (cap.isOpened()):
      print("Could not open video device")
  while True: 
      ret,frame= cap.read()
      frame = detector.findHands(frame)
      lmList = detector.findPosition(frame)
      if len(lmList)!=0:  
        print(lmList[8])

      currTime = time.time()
      fps = 1/(currTime-prevTime)
      prevTime = currTime

      cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
                  (255, 255, 255), 3)

      cv2.imshow("Live",frame)
      cv2.waitKey(1)
  cv2.destroyAllWindows()


if __name__ == "__main__":
    main()