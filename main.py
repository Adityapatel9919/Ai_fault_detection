"""
=========================================================
AI-Based Real-Time Fault Detection System
Main Application

Author : Aditya Patel
=========================================================
"""

from src.protection.protection_manager import ProtectionManager
from src.utils.logger import startup, shutdown


def main():

    startup()

    print("\nAI-Based Transmission Line Protection System Started\n")

    # ---------------------------------------------
    # Initialize Protection Manager
    # ---------------------------------------------

    manager = ProtectionManager(
        line_length_km=10
    )

    # ---------------------------------------------
    # Example Input
    # Replace these with Arduino readings later
    # ---------------------------------------------

    va = 230.0
    vb = 229.0
    vc = 231.0

    ia = 5.20
    ib = 5.10
    ic = 5.00

    # ---------------------------------------------
    # Run Protection Cycle
    # ---------------------------------------------

    result = manager.process(
        va,
        vb,
        vc,
        ia,
        ib,
        ic
    )

    # ---------------------------------------------
    # Display Results
    # ---------------------------------------------

    print("=" * 60)
    print("AI PROTECTION RESULT")
    print("=" * 60)

    print("\nPrediction")
    for key, value in result["Prediction"].items():
        print(f"{key:20}: {value}")

    print("\nFault Information")
    for key, value in result["Fault Information"].items():
        print(f"{key:20}: {value}")

    print("\nFault Location")
    for key, value in result["Location"].items():
        print(f"{key:20}: {value}")

    print("\nRelay")
    for key, value in result["Relay"].items():
        print(f"{key:20}: {value}")

    print("=" * 60)

    shutdown()


if __name__ == "__main__":
    main()