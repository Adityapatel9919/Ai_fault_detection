"""
=========================================================
Symmetrical Component Calculation
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

import numpy as np

# ==========================================================
# ROTATION OPERATOR
# ==========================================================

# a = e^(j120)

a = np.exp(1j * 2 * np.pi / 3)

A = np.array([

    [1, 1, 1],

    [1, a**2, a],

    [1, a, a**2]

], dtype=complex)

# Transformation Matrix

T = (1 / 3) * A


# ==========================================================
# VOLTAGE SEQUENCE COMPONENTS
# ==========================================================

def voltage_sequence_components(
    va,
    vb,
    vc
):
    """
    Calculates

    V0
    V1
    V2
    """

    phase = np.array(
        [
            va,
            vb,
            vc
        ],
        dtype=complex
    )

    sequence = T @ phase

    V0 = sequence[0]

    V1 = sequence[1]

    V2 = sequence[2]

    return V0, V1, V2


# ==========================================================
# CURRENT SEQUENCE COMPONENTS
# ==========================================================

def current_sequence_components(
    ia,
    ib,
    ic
):
    """
    Calculates

    I0
    I1
    I2
    """

    phase = np.array(
        [
            ia,
            ib,
            ic
        ],
        dtype=complex
    )

    sequence = T @ phase

    I0 = sequence[0]

    I1 = sequence[1]

    I2 = sequence[2]

    return I0, I1, I2


# ==========================================================
# MAGNITUDES
# ==========================================================

def sequence_magnitudes(
    V0,
    V1,
    V2,
    I0,
    I1,
    I2
):
    """
    Returns magnitudes only.
    """

    return {

        "V0": abs(V0),
        "V1": abs(V1),
        "V2": abs(V2),

        "I0": abs(I0),
        "I1": abs(I1),
        "I2": abs(I2)

    }


# ==========================================================
# UNBALANCE RATIOS
# ==========================================================

def sequence_ratios(
    V0,
    V1,
    V2,
    I0,
    I1,
    I2
):
    """
    Useful ML features.
    """

    eps = 1e-6

    return {

        "Voltage_Negative_Ratio":

            abs(V2) / (abs(V1) + eps),

        "Voltage_Zero_Ratio":

            abs(V0) / (abs(V1) + eps),

        "Current_Negative_Ratio":

            abs(I2) / (abs(I1) + eps),

        "Current_Zero_Ratio":

            abs(I0) / (abs(I1) + eps)

    }


# ==========================================================
# FAULT INDICES
# ==========================================================

def fault_indices(
    V0,
    V1,
    V2,
    I0,
    I1,
    I2
):
    """
    Returns useful protection indices.
    """

    eps = 1e-6

    return {

        "Voltage_Unbalance":

            abs(V2) / (abs(V1) + eps),

        "Current_Unbalance":

            abs(I2) / (abs(I1) + eps),

        "Ground_Fault_Index":

            abs(I0),

        "Negative_Sequence_Index":

            abs(I2)

    }


# ==========================================================
# COMPLETE FEATURE EXTRACTION
# ==========================================================

def extract_sequence_features(
    va,
    vb,
    vc,
    ia,
    ib,
    ic
):

    V0, V1, V2 = voltage_sequence_components(
        va,
        vb,
        vc
    )

    I0, I1, I2 = current_sequence_components(
        ia,
        ib,
        ic
    )

    features = {}

    features.update(

        sequence_magnitudes(

            V0,
            V1,
            V2,

            I0,
            I1,
            I2

        )

    )

    features.update(

        sequence_ratios(

            V0,
            V1,
            V2,

            I0,
            I1,
            I2

        )

    )

    features.update(

        fault_indices(

            V0,
            V1,
            V2,

            I0,
            I1,
            I2

        )

    )

    return features