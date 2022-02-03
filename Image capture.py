import cv2
import numpy as np
import time
pedestrian_detect = cv2.CascadeClassifier('haarcascade_fullbody.xml')
capture = cv2.VideoCapture('cctv.avi')
while capture.isOpened():
      time.sleep(.05)

      ret, frame = capture.read()

      frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
      pedestrian_detected = pedestrian_detect.detectMultiScale(gray, 1.2, 3)
      for (x,y,w,h) in pedestrian_detected:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
        cv2.putText(frame, "Person",(int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),
                    2, cv2.LINE_AA)
        cv2.imshow('Pedestrian Detection', frame)
      c = cv2.waitKey(1)
      if c == 27:
            break
capture.release()
cv2.destroyAllWindows()
