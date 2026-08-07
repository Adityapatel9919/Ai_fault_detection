"""
=========================================================
Fault Locator Module
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

import numpy as np

from config.config import LINE_IMPEDANCE_PER_KM


# ==========================================================
# Phase Impedance
# ==========================================================

def calculate_phase_impedance(voltage, current):
    """
    Calculate phase impedance.

    Parameters
    ----------
    voltage : float
    current : float

    Returns
    -------
    complex
    """

    if abs(current) < 1e-6:
        return complex(0, 0)

    return voltage / current


# ==========================================================
# Fault Impedance
# ==========================================================

def calculate_fault_impedance(
    va,
    vb,
    vc,
    ia,
    ib,
    ic,
    fault_type
):
    """
    Calculate apparent fault impedance based on fault type.
    """

    fault_type = fault_type.upper()

    if fault_type == "AG":
        return calculate_phase_impedance(va, ia)

    elif fault_type == "BG":
        return calculate_phase_impedance(vb, ib)

    elif fault_type == "CG":
        return calculate_phase_impedance(vc, ic)

    elif fault_type == "AB":
        return calculate_phase_impedance(
            va - vb,
            ia - ib
        )

    elif fault_type == "BC":
        return calculate_phase_impedance(
            vb - vc,
            ib - ic
        )

    elif fault_type == "AC":
        return calculate_phase_impedance(
            va - vc,
            ia - ic
        )

    elif fault_type == "ABG":
        return (
            calculate_phase_impedance(va, ia) +
            calculate_phase_impedance(vb, ib)
        ) / 2

    elif fault_type == "BCG":
        return (
            calculate_phase_impedance(vb, ib) +
            calculate_phase_impedance(vc, ic)
        ) / 2

    elif fault_type == "ACG":
        return (
            calculate_phase_impedance(va, ia) +
            calculate_phase_impedance(vc, ic)
        ) / 2

    elif fault_type in ["ABC", "ABCG"]:
        za = calculate_phase_impedance(va, ia)
        zb = calculate_phase_impedance(vb, ib)
        zc = calculate_phase_impedance(vc, ic)

        return (za + zb + zc) / 3

    return complex(0, 0)


# ==========================================================
# Fault Distance
# ==========================================================

def calculate_fault_distance(
    fault_impedance,
    line_impedance_per_km=LINE_IMPEDANCE_PER_KM
):
    """
    Estimate distance to fault (km).
    """

    if abs(line_impedance_per_km) < 1e-6:
        return 0.0

    distance = abs(fault_impedance) / abs(line_impedance_per_km)

    return round(distance, 2)


# ==========================================================
# Fault Percentage
# ==========================================================

def calculate_fault_percentage(
    distance_km,
    line_length_km
):
    """
    Percentage location of fault.
    """

    if line_length_km <= 0:
        return 0.0

    return round(
        (distance_km / line_length_km) * 100,
        2
    )


# ==========================================================
# Protection Zone
# ==========================================================

def classify_protection_zone(
    distance_km,
    line_length_km
):
    """
    Zone-1 : 0-80%
    Zone-2 : 80-120%
    Zone-3 : >120%
    """

    percentage = calculate_fault_percentage(
        distance_km,
        line_length_km
    )

    if percentage <= 80:
        return "Zone-1"

    elif percentage <= 120:
        return "Zone-2"

    else:
        return "Zone-3"


# ==========================================================
# Fault Report
# ==========================================================

def generate_fault_report(
    va,
    vb,
    vc,
    ia,
    ib,
    ic,
    fault_type,
    line_length_km
):
    """
    Generate complete fault report.
    """

    z_fault = calculate_fault_impedance(
        va,
        vb,
        vc,
        ia,
        ib,
        ic,
        fault_type
    )

    distance = calculate_fault_distance(
        z_fault
    )

    percentage = calculate_fault_percentage(
        distance,
        line_length_km
    )

    zone = classify_protection_zone(
        distance,
        line_length_km
    )

    return {

        "Fault Type": fault_type,

        "Fault Impedance (Ohm)": round(abs(z_fault), 4),

        "Distance (km)": distance,

        "Fault Percentage": percentage,

        "Protection Zone": zone

    }


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    report = generate_fault_report(

        va=230,

        vb=229,

        vc=231,

        ia=5.0,

        ib=5.1,

        ic=4.9,

        fault_type="AG",

        line_length_km=10

    )

    print("\nFault Report\n")

    for key, value in report.items():
        print(f"{key}: {value}")