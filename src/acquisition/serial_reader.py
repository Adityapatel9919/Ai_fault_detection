"""
=========================================================
Serial Reader
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

import serial
import time

from config.config import (
    SERIAL_PORT,
    BAUD_RATE,
    SERIAL_TIMEOUT
)

from src.utils.logger import info, error


class SerialReader:
    """
    Reads voltage and current measurements from Arduino.
    Expected format:
    Va,Vb,Vc,Ia,Ib,Ic
    """

    def __init__(
        self,
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        timeout=SERIAL_TIMEOUT
    ):

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    # ======================================================
    # Connect
    # ======================================================

    def connect(self):

        try:

            self.ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout
            )

            time.sleep(2)

            info(f"Connected to {self.port}")

            return True

        except Exception as e:

            error(f"Connection Failed: {e}")

            return False

    # ======================================================
    # Disconnect
    # ======================================================

    def disconnect(self):

        if self.ser and self.ser.is_open:

            self.ser.close()

            info("Serial connection closed.")

    # ======================================================
    # Read Sample
    # ======================================================

    def read_sample(self):

        if self.ser is None:

            return None

        try:

            line = self.ser.readline().decode("utf-8").strip()

            if not line:

                return None

            values = line.split(",")

            if len(values) != 6:

                return None

            return {

                "Va": float(values[0]),
                "Vb": float(values[1]),
                "Vc": float(values[2]),

                "Ia": float(values[3]),
                "Ib": float(values[4]),
                "Ic": float(values[5])

            }

        except Exception as e:

            error(f"Read Error: {e}")

            return None

    # ======================================================
    # Flush Buffer
    # ======================================================

    def flush(self):

        if self.ser:

            self.ser.reset_input_buffer()

    # ======================================================
    # Connection Status
    # ======================================================

    def is_connected(self):

        return self.ser is not None and self.ser.is_open


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    reader = SerialReader()

    if reader.connect():

        print("Reading data...\n")

        while True:

            sample = reader.read_sample()

            if sample:

                print(sample)

    else:

        print("Unable to connect.")