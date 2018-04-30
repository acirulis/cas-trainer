import cv2

vidcap = cv2.VideoCapture('data/live.mp4');
success, image = vidcap.read()
count = 0
success = True

while success:
    success, image = vidcap.read()
    print('read a new frame:', success)
    if count % 10 == 0:
        #420 / 680 C1
        # 1500 / 1040
        image_cropped = image[680:1040, 420:1500]
        cv2.imwrite('data/frames/frame%d.jpg' % count, image_cropped)
        print('success')
    count += 1
