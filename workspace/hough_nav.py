import urllib.request
import cv2 as cv
import numpy as np
import serial
import time
import math

# Define the Serial Port to Arduino & Baud Rate
ser = serial.Serial('/dev/cu.ESP32test-ESP32SPP', 115200)

# Define the ip address for video stream
url = 'http://192.168.4.1/cam-lo.jpg'

while True:
    # Open video stream address
    imgResp = urllib.request.urlopen(url)

    # Image Pre-Processing
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img = cv.imdecode(imgNp,-1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5) ,0)
    crop_img = blur[160:240, 0:320]

    # Canny Edge Detection and Hough Transform
    edges = cv.Canny(crop_img, 100, 200)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=250)

    # Sketching the Detected lines
    if lines is None:
        print('No lines detected', end="               ")
        ser.write(b'h')
    else:
        x1_sum = 0
        y1_sum = 0
        x2_sum = 0
        y2_sum = 0
        div = 0
        grad = 0.0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1_sum += x1
            y1_sum += y1
            x2_sum += x2
            y2_sum += y2
            slop = math.atan2((y2-y1), (x2-x1))
            grad += slop
            div += 1
    
        x1_sum /= div
        y1_sum /= div
        x2_sum /= div
        y2_sum /= div
        x1_sum = int(x1_sum)
        y1_sum = int(y1_sum)
        x2_sum = int(x2_sum)
        y2_sum = int(y2_sum)
        grad /= div
        grad_deg = (grad*180/np.pi)%360
        cv.arrowedLine(crop_img, (x1_sum, y1_sum), (x2_sum, y2_sum), (255, 0, 0), 3)

        # Behaviour Determination
        if(grad_deg >= 180.0):
            grad_deg -= 180.0
    
        print('Avg. Angle: ', grad_deg)
    
        if((grad_deg < 115.0) and (grad_deg > 45.0)):
            print('Go Straight')
            ser.write(b'w')
        elif(grad_deg >=115.0):
            print('Turn Right')
            ser.write(b'd')
        elif(grad_deg <= 45.0):
            print('Turn Left')
            ser.write(b'a')
        else:
            ser.write(b'h')

    # Display image
    cv.imshow('Estimation', crop_img)
    if ord('q')==cv.waitKey(10):
        break
