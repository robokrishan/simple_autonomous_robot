import serial
import time

# Define the serial port and baud rate.
# Ensure the 'COM#' corresponds to what was seen in the Port Monitor of your IDE
ser = serial.Serial('/dev/cu.usbserial-1410', 115200)

# System for converting string commands to char commands and writing to serial
# device ser

def func():
    user_input = input("\n Type on / off / quit : ")
    if user_input =="center":
        print("FoV Centered")
        time.sleep(0.1) 
        ser.write(b'g') 
        func()
    elif user_input =="left":
        print("Tilting Left...")
        time.sleep(0.1)
        ser.write(b'l')
        func()
    elif user_input =="right":
        print("Tilting Right...")
        time.sleep(0.1)
        ser.write(b'r')
        func()
    elif user_input =="up":
        print("Panning Up...")
        time.sleep(0.1)
        ser.write(b'u')
        func()
    elif user_input =="down":
        print("Panning Down...")
        time.sleep(0.1)
        ser.write(b'd')
        func()
    else:
        print("Invalid input. Type on / off / quit.")
        func()

time.sleep(2) # wait for the serial connection to initialize

func()
