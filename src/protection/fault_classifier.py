"""
=========================================================
Fault Classifier Module
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

from dataclasses import dataclass


# ==========================================================
# Fault Information
# ==========================================================

@dataclass
class FaultInfo:

    fault_type: str

    category: str

    severity: str

    phases: list

    ground_fault: bool


# ==========================================================
# Fault Classifier
# ==========================================================

class FaultClassifier:

    def __init__(self):

        self.database = {

            "Normal": FaultInfo(
                "Normal",
                "Healthy",
                "None",
                [],
                False
            ),

            "AG": FaultInfo(
                "AG",
                "Single Line to Ground",
                "Medium",
                ["A"],
                True
            ),

            "BG": FaultInfo(
                "BG",
                "Single Line to Ground",
                "Medium",
                ["B"],
                True
            ),

            "CG": FaultInfo(
                "CG",
                "Single Line to Ground",
                "Medium",
                ["C"],
                True
            ),

            "AB": FaultInfo(
                "AB",
                "Line to Line",
                "High",
                ["A", "B"],
                False
            ),

            "BC": FaultInfo(
                "BC",
                "Line to Line",
                "High",
                ["B", "C"],
                False
            ),

            "AC": FaultInfo(
                "AC",
                "Line to Line",
                "High",
                ["A", "C"],
                False
            ),

            "ABG": FaultInfo(
                "ABG",
                "Double Line to Ground",
                "Very High",
                ["A", "B"],
                True
            ),

            "BCG": FaultInfo(
                "BCG",
                "Double Line to Ground",
                "Very High",
                ["B", "C"],
                True
            ),

            "ACG": FaultInfo(
                "ACG",
                "Double Line to Ground",
                "Very High",
                ["A", "C"],
                True
            ),

            "ABC": FaultInfo(
                "ABC",
                "Three Phase",
                "Critical",
                ["A", "B", "C"],
                False
            ),

            "ABCG": FaultInfo(
                "ABCG",
                "Three Phase to Ground",
                "Critical",
                ["A", "B", "C"],
                True
            )

        }

    # ======================================================
    # Get Fault Information
    # ======================================================

    def classify(self, fault_type):

        return self.database.get(

            fault_type,

            FaultInfo(
                "Unknown",
                "Unknown",
                "Unknown",
                [],
                False
            )

        )

    # ======================================================
    # Check Ground Fault
    # ======================================================

    def is_ground_fault(self, fault_type):

        return self.classify(

            fault_type

        ).ground_fault

    # ======================================================
    # Severity
    # ======================================================

    def severity(self, fault_type):

        return self.classify(

            fault_type

        ).severity

    # ======================================================
    # Category
    # ======================================================

    def category(self, fault_type):

        return self.classify(

            fault_type

        ).category

    # ======================================================
    # Report
    # ======================================================

    def generate_report(self, fault_type):

        info = self.classify(

            fault_type

        )

        return {

            "Fault": info.fault_type,

            "Category": info.category,

            "Severity": info.severity,

            "Affected Phases": info.phases,

            "Ground Fault": info.ground_fault

        }


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    classifier = FaultClassifier()

    report = classifier.generate_report("ABG")

    print("\nFault Report\n")

    for key, value in report.items():

        print(f"{key}: {value}")