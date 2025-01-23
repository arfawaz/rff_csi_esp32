
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:53:57 2025

@author: fawaz
"""

'''
This code is intended to check if the device is recognized and if a serial 
connection can be successfully established with the specified port and baud rate.
Once you get the message "Successfully connect to port{port name}" it will start 
printing received data which might not make no sense. You can stop running the
code manually after this step.
'''

import serial  # Import the pySerial module for serial communication
import time  # Import the time module to add delays

# Define the serial port and baud rate (adjust these based on your device configuration)
port = '/dev/ttyUSB0'  # Update with your actual port (e.g., COM3 on Windows)
baudrate = 115200  # Communication speed in bits per second

try:
    # Attempt to establish a serial connection
    ser = serial.Serial(port, baudrate)  # Open the serial port with the specified parameters
    print(f"Successfully connected to {port} at {baudrate} baud.")  # Confirmation message
    
    time.sleep(2)  # Wait 2 seconds for the ESP32 (or other device) to initialize properly
    
    # Enter an infinite loop to continuously read data
    while True:
        # Check if there's data waiting in the serial buffer
        if ser.in_waiting > 0:  
            # Read all the available data from the buffer
            data = ser.read(ser.in_waiting)
            # Decode the byte data to a string and print it
            print(data.decode(errors='ignore'))  # Replace `decode()` with `decode(errors='ignore')` if you want to ignore invalid characters
            
except serial.SerialException as e:  
    # Handle errors related to the serial connection (e.g., port not available)
    print(f"Error: {e}")
    
finally:
    # Close the serial connection when the program ends
    if 'ser' in locals() and ser.is_open:
        ser.close()
