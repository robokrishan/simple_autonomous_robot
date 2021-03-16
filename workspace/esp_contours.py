import urllib.request
import cv2
import numpy as np
import serial
import time

# Define the Serial Port to Arduino & Baud Rate
#ser = serial.Serial('/dev/cu.ESP32test-ESP32SPP', 115200)

# Define the ip address for video stream
url = 'http://192.168.4.1/cam-lo.jpg'

while True:

    # Open video stream address
    imgResp = urllib.request.urlopen(url)

    # Convert to numpy byte array
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)

    # Decode image with opencv
    img = cv2.imdecode(imgNp,-1)

    # All image processing done here
    crop_img = img[160:240, 0:320]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)      # convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5) ,0)
    ret, thresh = cv2.threshold(blur,60,255,cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh.copy(), 1, cv2.CHAIN_APPROX_NONE)

     # Find the biggest contour (if detected)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        cv2.line(crop_img,(cx,0),(cx,720),(255,0,0),1)
        cv2.line(crop_img,(0,cy),(1280,cy),(255,0,0),1)
        cv2.drawContours(crop_img, contours, -1, (0,255,0), 1)

        if cx >= 240:
            print ("Turn Right!")
            #ser.write(b'l')
        if cx < 240 and cx > 80:
            print ("On Track!")
            #ser.write(b'w')
        if cx <= 80:
            print ("Turn Left!")
            #ser.write(b'r')
            
    else:
        print ("I don't see the line")
        #ser.write(b'h')

    # Display image
    cv2.imshow('line', img)
    if ord('q')==cv2.waitKey(10):
        break
