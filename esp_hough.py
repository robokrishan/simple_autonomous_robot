import urllib.request
import cv2
import numpy as np
import serial
import time

'''
        Actuation Control achieved through serial communication
        via NRF24L01+ WiFi module connected to Arduino Mega.

        List of commands -
            - Forward -> w
            - Reverse -> s
            - Turn Left -> a
            - Turn Right -> d
            - Pan Up -> u
            - Pan Down -> j
            - Tilt Left -> l
            - Tilt right -> r
'''

# Define the Serial Port to Arduino & Baud Rate
ser = serial.Serial('/dev/cu.ESP32test-ESP32SPP', 115200)

# Define the ip address for video stream
url = 'http://192.168.4.1/cam-lo.jpg'

command = ""

while True:

    # Open video stream address
    imgResp = urllib.request.urlopen(url)

    # Convert to numpy byte array
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)

    # Decode image with opencv
    img = cv2.imdecode(imgNp,-1)
    
    # All image processing done here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5) ,0)

    edges = cv2.Canny(blur,100,200)

    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=10, maxLineGap=250)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 10)
    theta_sum = 0.0
    div = 0
    if lines is None:
        print('No Line Detected')
    else:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
            theta_sum = theta_sum + theta
            div = div+1

        theta_sum /= div
        print('Angle: ', theta_sum)
        theta_sum = theta_sum * 180 / np.pi
        ser.write(theta_sum)

        
    


    # Display image
    cv2.imshow('line', img)
    if ord('q')==cv2.waitKey(10):
        break

