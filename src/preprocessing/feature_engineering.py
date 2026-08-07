"""
=========================================================
Feature Engineering Module
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

import numpy as np
import pandas as pd

# ==========================================================
# BASIC FEATURES
# ==========================================================

def average_voltage(va, vb, vc):
    return (va + vb + vc) / 3


def average_current(ia, ib, ic):
    return (ia + ib + ic) / 3


# ==========================================================
# RMS VALUES
# ==========================================================

def rms(signal):
    signal = np.asarray(signal)
    return np.sqrt(np.mean(np.square(signal)))


# ==========================================================
# PHASE IMPEDANCES
# ==========================================================

def phase_impedance(v, i):

    if abs(i) < 1e-6:
        return 0

    return v / i


# ==========================================================
# AVERAGE IMPEDANCE
# ==========================================================

def average_impedance(za, zb, zc):
    return (za + zb + zc) / 3


# ==========================================================
# VOLTAGE IMBALANCE
# ==========================================================

def voltage_imbalance(va, vb, vc):

    avg = average_voltage(va, vb, vc)

    maximum = max(
        abs(va - avg),
        abs(vb - avg),
        abs(vc - avg)
    )

    return maximum / avg


# ==========================================================
# CURRENT IMBALANCE
# ==========================================================

def current_imbalance(ia, ib, ic):

    avg = average_current(ia, ib, ic)

    if avg == 0:
        return 0

    maximum = max(
        abs(ia - avg),
        abs(ib - avg),
        abs(ic - avg)
    )

    return maximum / avg


# ==========================================================
# VOLTAGE DIFFERENCES
# ==========================================================

def voltage_difference(va, vb, vc):

    return (

        abs(va - vb),

        abs(vb - vc),

        abs(vc - va)

    )


# ==========================================================
# CURRENT DIFFERENCES
# ==========================================================

def current_difference(ia, ib, ic):

    return (

        abs(ia - ib),

        abs(ib - ic),

        abs(ic - ia)

    )


# ==========================================================
# APPARENT POWER
# ==========================================================

def apparent_power(v, i):

    return v * i


# ==========================================================
# ACTIVE POWER
# ==========================================================

def active_power(v, i, pf=1):

    return v * i * pf


# ==========================================================
# REACTIVE POWER
# ==========================================================

def reactive_power(v, i, pf=1):

    angle = np.arccos(np.clip(pf, -1, 1))

    return v * i * np.sin(angle)


# ==========================================================
# POWER FACTOR
# ==========================================================

def power_factor(active, apparent):

    if apparent == 0:
        return 0

    return active / apparent


# ==========================================================
# RATE OF CHANGE
# ==========================================================

def rate_of_change(signal):

    signal = np.asarray(signal)

    if len(signal) < 2:
        return 0

    return np.mean(np.diff(signal))


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features(row):
    """
    Input:
    One row of dataset

    Output:
    Feature Dictionary
    """

    va = row["Va"]
    vb = row["Vb"]
    vc = row["Vc"]

    ia = row["Ia"]
    ib = row["Ib"]
    ic = row["Ic"]

    # -------------------------
    # Average Values
    # -------------------------

    avg_v = average_voltage(
        va,
        vb,
        vc
    )

    avg_i = average_current(
        ia,
        ib,
        ic
    )

    # -------------------------
    # Phase Impedance
    # -------------------------

    za = phase_impedance(va, ia)

    zb = phase_impedance(vb, ib)

    zc = phase_impedance(vc, ic)

    avg_z = average_impedance(
        za,
        zb,
        zc
    )

    # -------------------------
    # Voltage Imbalance
    # -------------------------

    v_imb = voltage_imbalance(
        va,
        vb,
        vc
    )

    # -------------------------
    # Current Imbalance
    # -------------------------

    i_imb = current_imbalance(
        ia,
        ib,
        ic
    )

    # -------------------------
    # Voltage Difference
    # -------------------------

    vab, vbc, vca = voltage_difference(
        va,
        vb,
        vc
    )

    # -------------------------
    # Current Difference
    # -------------------------

    iab, ibc, ica = current_difference(
        ia,
        ib,
        ic
    )

    # -------------------------
    # Power
    # -------------------------

    s = apparent_power(
        avg_v,
        avg_i
    )

    p = active_power(
        avg_v,
        avg_i
    )

    pf = power_factor(
        p,
        s
    )

    q = reactive_power(
        avg_v,
        avg_i,
        pf
    )

    return {

        "Va": va,
        "Vb": vb,
        "Vc": vc,

        "Ia": ia,
        "Ib": ib,
        "Ic": ic,

        "Average_Voltage": avg_v,
        "Average_Current": avg_i,

        "Za": za,
        "Zb": zb,
        "Zc": zc,

        "Average_Impedance": avg_z,

        "Voltage_Imbalance": v_imb,

        "Current_Imbalance": i_imb,

        "Vab": vab,
        "Vbc": vbc,
        "Vca": vca,

        "Iab": iab,
        "Ibc": ibc,
        "Ica": ica,

        "Apparent_Power": s,

        "Active_Power": p,

        "Reactive_Power": q,

        "Power_Factor": pf

    }


# ==========================================================
# DATAFRAME PROCESSING
# ==========================================================

def build_feature_dataframe(df):
    """
    Converts raw dataset into
    ML-ready feature dataset.
    """

    feature_rows = []

    for _, row in df.iterrows():

        feature_rows.append(

            extract_features(row)

        )

    return pd.DataFrame(feature_rows)