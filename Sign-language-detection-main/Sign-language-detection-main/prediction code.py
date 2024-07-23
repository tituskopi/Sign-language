import cv2, openpyxl, time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
from PIL import Image
from PIL import ImageDraw



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, minTrackCon=0.7)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
Feedback = ""

timer = 0
sr = False
sg = False


labels = ["BAD", "BATHROOM", "GOOD", "I NEED HELP", "HI HOW ARE YOU", "NO", "PHONECALL", "I AM FINE THANKYOU", "I NEED WATER", "YES"]
fingers = 0




while True:
    imgbg = cv2.imread("bg.jpg")
    success, img = cap.read()
    imgOutput = img.copy()

    imgbg[203:683, 445:1085] = imgOutput

    hands, img = detector.findHands(img)


    if sg:

        if sr is False:
            timer = time.time() - initialTime
            cv2.putText(imgOutput, str(int(timer)), (10,90), cv2.FONT_HERSHEY_PLAIN, 6, (255,0,255), 4)



        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            # print(fingers)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                if timer>1:
                    sr = True
                    timer = 0
                    Feedback = (labels[index])
                    obj = Feedback
                    engine = pyttsx3.init()
                    engine.say(obj)
                    engine.runAndWait()
                    engine.stop()
                    print(Feedback)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                  (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        cv2.rectangle(imgOutput, (x - offset, y - offset),
                  (x + w + offset, y + h + offset), (255, 0, 255), 4)
    #
    #
    #
    cv2.imshow("Image", imgOutput)
    imgbg = cv2.putText(imgbg, Feedback, (700, 800), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 3, cv2.LINE_AA)
    cv2.imshow("", imgbg)
    key = cv2.waitKey(1)


    #
    #
    if key == ord('s'):
        sg = True
        initialTime = time.time()
        sr = False




