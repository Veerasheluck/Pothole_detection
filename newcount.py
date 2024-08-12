import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *




model=YOLO('best.pt')
model.train


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('pothole1_Trim.mp4')


my_file = open("class.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
tracker=Tracker()   
area=[(0,266),(0,350),(483,350),(478,266)]
area_c=set()
area1=[(478,266),(483,350),(1017,350),(1013,266)]
area_c1=set()





while True:    
    ret,frame = cap.read()
    if not ret:
        break
#    count += 1
#    if count % 3 != 0:
#        continue


    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
#        if 'person' in c:
        list.append([x1,y1,x2,y2])
            
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        
        results=cv2.pointPolygonTest(np.array(area,np.int32),((x4,y4)),False)
        if results>=0:
           cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
           cv2.putText(frame,str(int(id)),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
           area_c.add(id)
        results1=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
        if results1>=0:
           cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
           cv2.putText(frame,str(int(id)),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
           area_c1.add(id)   
        
    area1_c=(len(area_c))
    area2_c=(len(area_c1))    
    #cv2.putText(frame,"left:"+str(int(area1_c)),(66,51),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),3)
    #cv2.putText(frame,"right:"+str(int(area2_c)),(883,51),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),3)

    a=area1_c+area2_c
    print(a)
    cv2.putText(frame,"TOTAL PITS :"+str(int(area1_c)),(66,51),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)


    
    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,0),2)
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()






