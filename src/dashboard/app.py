"""
=========================================================
Streamlit Dashboard
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

import streamlit as st

from src.protection.protection_manager import ProtectionManager

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(

    page_title="AI Fault Detection System",

    page_icon="⚡",

    layout="wide"

)

# ==========================================================
# Protection Manager
# ==========================================================

manager = ProtectionManager()

# ==========================================================
# Title
# ==========================================================

st.title("⚡ AI-Based Real-Time Fault Detection System")

st.markdown("---")

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("Input Measurements")

va = st.sidebar.number_input(

    "Phase-A Voltage (V)",

    value=230.0

)

vb = st.sidebar.number_input(

    "Phase-B Voltage (V)",

    value=229.0

)

vc = st.sidebar.number_input(

    "Phase-C Voltage (V)",

    value=231.0

)

ia = st.sidebar.number_input(

    "Phase-A Current (A)",

    value=5.2

)

ib = st.sidebar.number_input(

    "Phase-B Current (A)",

    value=5.1

)

ic = st.sidebar.number_input(

    "Phase-C Current (A)",

    value=5.0

)

# ==========================================================
# Run Prediction
# ==========================================================

if st.sidebar.button("Run Protection"):

    result = manager.process(

        va,

        vb,

        vc,

        ia,

        ib,

        ic

    )

    prediction = result["Prediction"]

    fault = result["Fault Information"]

    location = result["Location"]

    relay = result["Relay"]

    # -----------------------------------------
    # Prediction
    # -----------------------------------------

    st.header("Prediction")

    col1, col2 = st.columns(2)

    col1.metric(

        "Detected Fault",

        prediction["fault"]

    )

    col2.metric(

        "Confidence",

        f"{prediction['confidence']:.2%}"

    )

    # -----------------------------------------
    # Fault Information
    # -----------------------------------------

    st.header("Fault Information")

    st.json(fault)

    # -----------------------------------------
    # Fault Location
    # -----------------------------------------

    st.header("Fault Location")

    st.json(location)

    # -----------------------------------------
    # Relay
    # -----------------------------------------

    st.header("Relay Status")

    if relay["trip"]:

        st.error("⚠ Relay TRIPPED")

    else:

        st.success("✅ System Healthy")

    st.json(relay)

st.markdown("---")

st.caption("AI-Based Real-Time Fault Detection & Location using SVM + Impedance Method")