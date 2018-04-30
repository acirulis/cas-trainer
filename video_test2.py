import cv2

video = cv2.VideoCapture('data/live.mp4')
import time
time.sleep(1)
ret, frame = video.read()

cv2.imwrite('out2.jpeg', frame)

video.release()