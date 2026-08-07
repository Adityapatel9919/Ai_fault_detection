"""
=========================================================
Protection Manager
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

from src.models.predict import predict_fault

from src.protection.relay_logic import ProtectionRelay

from src.protection.fault_locator import (
    generate_fault_report
)

from src.protection.fault_classifier import (
    FaultClassifier
)

from src.utils.logger import (
    log_fault,
    info
)


class ProtectionManager:

    def __init__(

        self,

        line_length_km=10

    ):

        self.line_length = line_length_km

        self.relay = ProtectionRelay()

        self.classifier = FaultClassifier()

    # =====================================================
    # Main Protection Cycle
    # =====================================================

    def process(

        self,

        va,
        vb,
        vc,

        ia,
        ib,
        ic

    ):

        # --------------------------------------------
        # AI Prediction
        # --------------------------------------------

        prediction = predict_fault(

            va,
            vb,
            vc,

            ia,
            ib,
            ic

        )

        fault = prediction["fault"]

        confidence = prediction["confidence"]

        # --------------------------------------------
        # Fault Information
        # --------------------------------------------

        fault_info = self.classifier.generate_report(

            fault

        )

        # --------------------------------------------
        # Relay Logic
        # --------------------------------------------

        relay = self.relay.process(

            fault,

            confidence

        )

        # --------------------------------------------
        # Fault Location
        # --------------------------------------------

        location = generate_fault_report(

            va,

            vb,

            vc,

            ia,

            ib,

            ic,

            fault,

            self.line_length

        )

        # --------------------------------------------
        # Logging
        # --------------------------------------------

        log_fault(

            va,

            vb,

            vc,

            ia,

            ib,

            ic,

            fault,

            confidence,

            location["Distance (km)"],

            relay["relay_state"]

        )

        # --------------------------------------------
        # Final Output
        # --------------------------------------------

        result = {

            "Prediction": prediction,

            "Fault Information": fault_info,

            "Location": location,

            "Relay": relay

        }

        info("Protection cycle completed.")

        return result


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    manager = ProtectionManager(

        line_length_km=10

    )

    result = manager.process(

        va=230,

        vb=229,

        vc=231,

        ia=5.2,

        ib=5.1,

        ic=5.0

    )

    print()

    print("="*60)

    print(result)

    print("="*60)