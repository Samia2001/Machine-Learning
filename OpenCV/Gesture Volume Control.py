import cv2
import time
import numpy as np
import mediapipe as mp
import handtrackingmodule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
ptime = 0 
ctime = 0 

cap = cv2.VideoCapture(0)
detector = htm.handdetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0.0, None)
minVol = volRange[0]
maxvol = volRange[1]
while cap.isOpened():
    
    success, img = cap.read()
    img = detector.findHands(img)
    lmList=detector.findposition(img,draw=False)
    if len(lmList)!=0:
        #print(lmList[4],lmList[8])
        x1,y1=lmList[4][1],lmList[4][2]
        x2,y2=lmList[8][1],lmList[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        length = math.hypot(x2-x1,y2-y1)
        if length<50:
            cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
        
        #print(length)
        #hand range 50-300
        #volume range -65 -0
        vol = np.interp(length,[12,110],[minVol,maxvol])
        volbar = np.interp(length,[12,110],[420,150])
        cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
        cv2.rectangle(img,(50,int(volbar)),(85,400),(0,255,0),cv2.FILLED)
        volume.SetMasterVolumeLevel(vol, None)
        volctr = np.interp(length,[50,150],[0,100])
        cv2.putText(img,f'{ int(volctr)} % ',(40,400),cv2.FONT_HERSHEY_COMPLEX,1,(102,0,204),3)
        


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

