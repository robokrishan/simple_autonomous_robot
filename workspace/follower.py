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
#print('connecting bluetooth port...')
#time.sleep(3)
ser.write(0)
# Define the ip address for video stream
url = 'http://192.168.4.1/cam-lo.jpg'
imgResp = urllib.request.urlopen(url)


while True:
    # Open video stream address
    imgResp = urllib.request.urlopen(url)

    # Convert to numpy byte array
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)

    # Decode image with opencv
    image = cv2.imdecode(imgNp,-1)	
    Blackline = cv2.inRange(image, (0,0,0), (60,60,60))	
    kernel = np.ones((3,3), np.uint8)
    Blackline = cv2.erode(Blackline, kernel, iterations=5)
    Blackline = cv2.dilate(Blackline, kernel, iterations=9)	
    img_blk,contours_blk, hierarchy_blk = cv2.findContours(Blackline.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours_blk) > 0:	 
        blackbox = cv2.minAreaRect(contours_blk[0])
        (x_min, y_min), (w_min, h_min), ang = blackbox
        
        if ang < -45:
            ang = 90 + ang
        if w_min < h_min and ang > 0:
            ang = (90-ang) * -1
        if w_min > h_min and ang < 0:
            ang = 90 + ang

        setpoint = 320
        error = int(x_min - setpoint)
        print('Offset: ', ang)
        ang = int(ang)
        ser.write(ang)
        box = cv2.boxPoints(blackbox)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(0,0,255),3)
        cv2.putText(image,str(ang),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image,str(error),(10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(image, (int(x_min),200 ), (int(x_min),250 ), (255,0,0),3)
        cv2.imshow("orginal with line", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

