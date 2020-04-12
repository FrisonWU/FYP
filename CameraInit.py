import cv2
import numpy as np
import pickle


cap = cv2.VideoCapture(0)
index = 0
while True:
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2G)
    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == 27:  ##ESC
        break
    elif cv2.waitKey(1) & 0xFF == ord('l'):
        cv2.imwrite("Cam_hori.jpg", frame)
        continue
    elif cv2.waitKey(1) & 0xFF == ord('p'):
        cv2.imwrite("Cam_Heigh.jpg", frame)
        continue


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()