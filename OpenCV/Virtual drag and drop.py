import cv2
from cvzone.HandTrackingModule import HandDetector
#import handtracking as htm

clr=(255,0,255)
class dragrect():
    def __init__(self,poscenter,size=(50,50)):
        self.poscenter=poscenter
        self.size = size
    def update(self,lmList1):
        cx,cy=self.poscenter
        w,h=self.size

        if cx-w//2<=lmList1[8][0]<=cx+w//2 and cy-h//2<=lmList1[8][1]<=cy+h//2:
            cx ,cy = lmList1[8][0],lmList1[8][1]
            self.poscenter=cx,cy
            clr = (0,0,255)
        else:
            clr = (255,0,255)
        return clr
    

rectlist=[]
colorDict = {}
for i in range(5):
    rectlist.append(dragrect([100+i*100,100]))
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8)
cx,cy,w,h = 50,50,100,100      
while cap.isOpened():
    success, img = cap.read()
    img=cv2.flip(img,1)
    hands,img = detector.findHands(img)
    lmList1=[]

    for rect in rectlist:
        colorDict[rect] = (255, 0, 255)
    
    if(hands): 
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        index_finger_tip = lmList1[8][1:]
        middle_finger_tip = lmList1[12][1:]
        distance, _, _ = detector.findDistance(index_finger_tip, middle_finger_tip, img)
        #print(distance)
       
        if(distance<50):
            for rect in rectlist:
                colorDict[rect]=rect.update(lmList1)
    for rect in rectlist:
        cx,cy=rect.poscenter
        w,h=rect.size
        c = colorDict[rect]
        cv2.rectangle(img,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),c,cv2.FILLED)
        
    
    if success:
        image = cv2.resize(img, (600,400))
        cv2.imshow('Camera', image)
        if cv2.waitKey(25) & 0xff == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
