import cv2
import numpy as np
import serial
import urllib.request
import objc

def print_debug():
    print('Center (X,Y):\t(',x_min,', ',y_min,')')
    print('Width:\t', int(w_min))
    print('Height:\t', int(h_min))
    print('Vehicle Offset: ', error, '\n')
'''
try:
    # Bluetooth Serial Port
    ser = serial.Serial('/dev/cu.ESP32test-ESP32SPP', 115200)
    print('Blueooth Connection Established')
except:
    # USB-UART Serial Port
    print('Bluetooth Failed')
    ser = serial.Serial('/dev/cu.SLAB_USBtoUART', 115200)
    print('USB-UART Connection Established')
'''
# Camera Address
url = 'http://192.168.178.55/cam-lo.jpg'
try:
    print('Searching for camera on local server...')
    imgResp = urllib.request.urlopen(url, timeout=5)
    print('Camera found!')
except:
    print('Camera not connected to local server')
    print('Checking for direct connection...')
    objc.loadBundle('CoreWLAN',
    bundle_path = '/System/Library/Frameworks/CoreWLAN.framework',
    module_globals = globals())
    iface = CWInterface.interface()
    networks, error = iface.scanForNetworksWithName_error_('bot_cam', None)
    network = networks.anyObject()
    success, error = iface.associateToNetwork_password_error_(network, '', None)
    url = 'http://192.168.4.1/cam-lo.jpg'
    print('Direct connection with camera established!')

while True:
    # Image Acquisition
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    src = cv2.imdecode(imgNp,-1)

    # Filter by Color
    Blackline = cv2.inRange(src, (0,0,0), (60,60,60))

    # Erode & Dilate for Noise Suppression
    kernel = np.ones((3,3), np.uint8)
    Blackline = cv2.erode(Blackline, kernel, iterations=5)
    Blackline = cv2.dilate(Blackline, kernel, iterations=9)

    # Detect Contours
    img_blk, contours_blk, hierarchy_blk = cv2.findContours(Blackline.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Make Rectangle around Contour
    try:
        blackbox = cv2.minAreaRect(contours_blk[0])
    except:
        print('No Path Detected')

    # Unparse dimension and orientation of Rectangle
    (x_min, y_min), (w_min, h_min), ang = blackbox

    if ang < -45:
        ang += 90
    if w_min < h_min and ang > 0:
        ang = (90-ang) * -1
    if w_min > h_min and ang < 0:
        ang += 90

    # Define Offset Reference Point
    setpoint = 160

    # Compute Translational Error
    error = int(x_min - setpoint)
    #ser.write(int(ang))

    # Draw Rectangle
    box = cv2.boxPoints(blackbox)
    box = np.int0(box)
    cv2.drawContours(src,[box],0,(0,255,0),2)

    # Print Errors to Video
    cv2.putText(src,str(int(ang)),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(src,str(int(error)),(10, 224), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.line(src, (int(x_min), setpoint), (int(x_min),200 ), (255,0,0),3)

    cv2.imshow('Blackline', src)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        objc.loadBundle('CoreWLAN',
        bundle_path = '/System/Library/Frameworks/CoreWLAN.framework',
        module_globals = globals())
        iface = CWInterface.interface()
        networks, error = iface.scanForNetworksWithName_error_('FRITZ!Box 7490', None)
        network = networks.anyObject()
        success, error = iface.associateToNetwork_password_error_(network, '21162010819665839679', None)
        break
