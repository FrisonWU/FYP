import cv2
import numpy as np
import pickle


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
index = 0
while True:
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        cv2.imwrite("target.jpg", frame)
        index = index + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()