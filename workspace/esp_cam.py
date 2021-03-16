import urllib.request
import cv2
import numpy as np
import serial
import time

# Define the Serial Port to Arduino & Baud Rate
#ser = serial.Serial('/dev/cu.usbserial-1410', 115200)

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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # convert to grayscale


    # Display image
    cv2.imshow('grayscale', img)
    if ord('q')==cv2.waitKey(10):
        break
