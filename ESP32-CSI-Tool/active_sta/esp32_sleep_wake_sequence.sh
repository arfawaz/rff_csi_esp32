#!/bin/bash

# ---------------------------------------------------------
# ESP32 Data Collection Script for Multiple Cycles
# This script automates the process of toggling between two
# ESP32 devices (AP1 and AP2), collecting CSI data through
# idf.py commands, and alternating between the devices to 
# collect data in a continuous cycle.
# The script allows you to control the ESP32 devices by 
# putting them in sleep mode and waking them up, as well as 
# handling the data collection process for multiple cycles.
# You can run the script by opening a terminal inside the active_sta folder and running the following command:
# ./script.sh 5 10 20 /dev/ttyUSB1 /dev/ttyUSB2 /dev/ttyUSB0

# ---------------------------------------------------------

# Set up the ESP-IDF environment (required for ESP32 programming)
# This script requires that the ESP-IDF environment is set up.
# It loads the necessary variables and configurations for 
# `idf.py` commands to work properly.

. $HOME/esp/esp-idf/export.sh


# ---------------------------------------------------------
# Check for required input parameters
# The script expects 6 parameters: the number of cycles, 
# the wait time between switching devices, data collection 
# time, and the USB ports for both ESP32 devices.
# If the correct number of parameters is not provided, the 
# script exits with an error message.
# ---------------------------------------------------------

if [ $# -ne 6 ]; then
  echo "Usage: $0 <n_cycles> <wait_time_in_seconds> <data_collection_time> <usb0_port> <usb1_port> <idf_usb_port>"
  exit 1
fi

# Assign input parameters to variables
N_CYCLES=$1                 # Number of cycles to repeat
WAIT_TIME=$2                # Wait time before switching APs
DATA_COLLECTION_TIME=$3     # Data collection duration
USB0=$4                     # USB port for AP1
USB1=$5                     # USB port for AP2
IDF_USB=$6                  # USB port used for idf.py command

# ---------------------------------------------------------
# Function to toggle sleep mode on an ESP32 device
# This function will send the sleep or wake signal to an 
# ESP32 device via its USB port by manipulating the DTR pin.
# The parameter $1 is the USB port, and $2 is the state:
# "True" for waking up the device, "False" for sleeping it.
# ---------------------------------------------------------

toggle_sleep() {
    local port=$1
    local state=$2  # True to wake up, False to sleep

    python3 -c "
import serial
import time
ser = serial.Serial('$port', 115200, timeout=1)
ser.dtr = $state  # False to sleep, True to wake up
time.sleep(2)  # Allow ESP32 time to process sleep/wake
ser.close()
"
}

# ---------------------------------------------------------
# Function to safely stop any running idf.py process
# This function looks for any running idf.py processes
# using `lsof`, then kills them to free up the USB port.
# This is crucial before reusing the port for the next cycle.
# ---------------------------------------------------------

stop_idf_monitor() {
    echo "Stopping any running idf.py process..."

    # Identify the process ID using lsof
    PID=$(sudo lsof -t "$IDF_USB")

    # If there is a PID, kill the process
    if [ -n "$PID" ]; then
        echo "Process ID $PID is using /dev/ttyUSB0. Terminating it..."
        sudo kill -9 $PID
    else
        echo "No process found using /dev/ttyUSB0."
    fi
}


# ---------------------------------------------------------
# Function to wait until a specific USB device is available
# This function checks for the availability of the USB port.
# It waits for the port to become available, retrying up to 15 times.
# ---------------------------------------------------------

wait_for_usb() {
    local usb_port=$1
    echo "Waiting for $usb_port to be available..."
    for i in {1..15}; do
        if [ -c "$usb_port" ]; then
            echo "$usb_port is now available."
            return
        fi
        sleep 1
    done
    echo "Warning: $usb_port did not become available!"
}

# ---------------------------------------------------------
# Function to start CSI data collection
# This function runs the `idf.py monitor` command on the 
# specified USB port and pipes the output to a Python script 
# that captures CSI data and saves it to a CSV file.
# It runs for the specified duration and then stops.
# ---------------------------------------------------------
start_data_collection() {
    echo "Starting data collection for $DATA_COLLECTION_TIME seconds..."
    idf.py -p "$IDF_USB" monitor | python3 /home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/utilities/capture_csi.py \
        /home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/rff_csi_esp32/csi_data_collected/02_feb_2025/csi_antenna_test.csv &
    COLLECT_PID=$!
    sleep $DATA_COLLECTION_TIME
    kill $COLLECT_PID
    echo "Data collection stopped."
}


# ---------------------------------------------------------
# Main loop to run the cycle n times
# The loop iterates for the number of cycles specified by
# the user. It alternates between waking up and putting to 
# sleep two ESP32 devices (AP1 and AP2), and collects data
# during each phase. The wait times and data collection times
# are controlled by the user input.
# ---------------------------------------------------------

for ((i=1; i<=N_CYCLES; i++)); do
    echo "🔄 Starting cycle $i..."

    # Step 1: Stop any running idf.py process
    stop_idf_monitor

    # Step 2: Put AP2 to sleep
    echo "💤 Putting AP2 to sleep..."
    toggle_sleep "$USB1" "False"
    echo "✅ AP2 is now asleep."

    # Step 3: Wake up AP1
    echo "🚀 Waking up AP1..."
    toggle_sleep "$USB0" "True"
    sleep 5  # Allow ESP32 time to wake up
    wait_for_usb "$USB0"
    echo "✅ AP1 is now awake."

    # Step 4: Wait before data collection
    echo "⏳ Waiting for $WAIT_TIME seconds..."
    sleep $WAIT_TIME

    # Step 5: Start STA1 data collection
    start_data_collection

    # Step 6: Stop idf.py before switching APs
    stop_idf_monitor

    # Step 7: Put AP1 to sleep
    echo "💤 Putting AP1 to sleep..."
    toggle_sleep "$USB0" "False"
    echo "✅ AP1 is now asleep."

    # Step 8: Wake up AP2
    echo "🚀 Waking up AP2..."
    toggle_sleep "$USB1" "True"
    sleep 5  # Allow ESP32 time to wake up
    wait_for_usb "$USB1"
    echo "✅ AP2 is now awake."

    # Step 9: Wait before next data collection
    echo "⏳ Waiting for another $WAIT_TIME seconds..."
    sleep $WAIT_TIME

    # Step 10: Start STA1 data collection
    start_data_collection

    # Step 11: Stop idf.py before ending the cycle
    stop_idf_monitor

    # Step 12: Put AP2 to sleep
    echo "💤 Putting AP2 to sleep..."
    toggle_sleep "$USB1" "False"
    echo "✅ AP2 is now asleep."

    echo "✅ Cycle $i completed."
done

echo "🎉 All $N_CYCLES cycles completed successfully!"
