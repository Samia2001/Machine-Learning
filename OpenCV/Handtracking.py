import cv2
import time
import numpy as np
import mediapipe as mp
class handdetector():
    def __init__(self, mode = False, maxHands = 2, modelC=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelC, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img
           
    def findposition(self,img,handNo=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return lmList  
     
def main():
    ptime = 0 
    ctime = 0 
  
    cap = cv2.VideoCapture(0)
    detector = handdetector()
    while cap.isOpened():
        
        success, img = cap.read()
        img = detector.findHands(img)
        lmList=detector.findposition(img,draw=False)
        # print(lmList[0])
        ctime = time.time()
        fps=1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img,f'FPS:{int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(102,0,204),3)
        if success:
            image = cv2.resize(img, (600,400))
            cv2.imshow('Camera', image)
            if cv2.waitKey(25) & 0xff == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    

main()
