import time
import cv2
from collections import deque
import math
import numpy as np

def imshow_autosize(winname, img):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(winname, cv2.WINDOW_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(winname, img)

def mosaic(img, alpha):
    # 高さと幅
    w = img.shape[1]
    h = img.shape[0]

    # モザイク加工
    img = cv2.resize(img, (int(w*alpha), int(h*alpha)))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    return img

mouthCascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
noseCascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# 直近N_SAMPLE分の時刻を保持
N_SAMPLE = 5
q = deque([time.time() for i in range(N_SAMPLE)])

wait = 0.1 # 処理負荷[sec]


t1 = 0

cap = cv2.VideoCapture(0)







while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #mouth = mouthCascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=40)
    faces = faceCascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
            # Draw a rectangle around the faces
    t = 0
    r = 0
    r1 = 0
    x1 = 0
    y1 = 0
    for (x, y, w, h) in faces:

        #顔囲い
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray_mouth = frame[y:y+h, x:x+w]

        roi_color_mouth = frame[y+(int(h/2)):y+h, x:x+w]
        a = x + (math.floor(float(w) / 2))
        b = y + (math.floor(float(h) / 2))
        c = (math.floor(float(w) / 2)) + (math.floor(float(w) / 5))
        t += 1
        r = 0


        mouth = mouthCascade.detectMultiScale(roi_color_mouth,scaleFactor=1.1,minNeighbors=2)
        for (ex,ey,ew,eh) in mouth:
            cv2.line(frame, (x,y), (x+w, y+h), (255,0,0), thickness=3, lineType=cv2.LINE_8)
            cv2.line(frame, (x+w,y), (x, y+h), (255,0,0), thickness=3, lineType=cv2.LINE_8)


            #未着用別窓表示
            #cv2.circle(frame, (a, b), c, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
            #out_face = frame[y:y+h, x:x+w]
            #imshow_autosize('out_face'+str(r),out_face)
            r +=1
            x1 = x
            y1 = y

        if len(mouth) == 0:
            nose = noseCascade.detectMultiScale(roi_gray_mouth,scaleFactor=1.1,minNeighbors=5)
            for (mx,my,mw,mh) in nose:
                pts = np.array(((x, y+h), (a, y), (x+w, y+h)))
                cv2.polylines(frame, [pts], True, (0, 255, 0), thickness=2)
                #cv2.circle(frame, (a, b), c, (0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
            if len(nose) == 0:
                cv2.circle(frame, (a, b), c, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)





        #モザイク
        #frame[y:y+h, x:x+w] = mosaic(frame[y:y+h, x:x+w], 0.05)


    #fps算出
    now = time.time()
    fps = N_SAMPLE / (now - q.popleft())
    q.append(now)

    cv2.putText(frame,'{:6.3f}fps'.format(fps),(10,20),cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0))
    imshow_autosize('video',frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
