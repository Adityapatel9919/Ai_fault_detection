"""
=========================================================
Relay Logic Module
AI-Based Real-Time Fault Detection System

Author : Aditya Patel
=========================================================
"""

from collections import deque
from enum import Enum
import time

from config.config import (
    CONFIDENCE_THRESHOLD,
    MAJORITY_WINDOW,
    MIN_FAULT_COUNT
)


# ==========================================================
# Relay State
# ==========================================================

class RelayState(Enum):

    IDLE = "IDLE"

    MONITORING = "MONITORING"

    TRIPPED = "TRIPPED"

    RESET = "RESET"


# ==========================================================
# Protection Relay
# ==========================================================

class ProtectionRelay:

    def __init__(

        self,

        confidence_threshold=CONFIDENCE_THRESHOLD,

        window_size=MAJORITY_WINDOW,

        trip_threshold=MIN_FAULT_COUNT,

        cooldown=5

    ):

        self.confidence_threshold = confidence_threshold

        self.window_size = window_size

        self.trip_threshold = trip_threshold

        self.cooldown = cooldown

        self.state = RelayState.IDLE

        self.predictions = deque(maxlen=window_size)

        self.trip_time = None

    # ======================================================
    # Reset Buffer
    # ======================================================

    def clear(self):

        self.predictions.clear()

    # ======================================================
    # Add Prediction
    # ======================================================

    def update(

        self,

        fault_type,

        confidence

    ):

        """
        Called every sampling instant.
        """

        # Low confidence → Normal

        if confidence < self.confidence_threshold:

            fault_type = "Normal"

        self.predictions.append(fault_type)

        self.state = RelayState.MONITORING

    # ======================================================
    # Majority Voting
    # ======================================================

    def majority_vote(self):

        if len(self.predictions) < self.window_size:

            return False, "Normal"

        votes = {}

        for p in self.predictions:

            votes[p] = votes.get(p, 0) + 1

        fault = max(votes, key=votes.get)

        count = votes[fault]

        if fault != "Normal" and count >= self.trip_threshold:

            return True, fault

        return False, "Normal"

    # ======================================================
    # Relay Decision
    # ======================================================

    def trip_decision(self):

        trip, fault = self.majority_vote()

        if trip:

            self.state = RelayState.TRIPPED

            self.trip_time = time.time()

            return {

                "trip": True,

                "fault": fault

            }

        return {

            "trip": False,

            "fault": "Normal"

        }

    # ======================================================
    # Reset Relay
    # ======================================================

    def reset(self):

        if self.state != RelayState.TRIPPED:

            return

        elapsed = time.time() - self.trip_time

        if elapsed >= self.cooldown:

            self.clear()

            self.state = RelayState.IDLE

            self.trip_time = None

    # ======================================================
    # Current State
    # ======================================================

    def status(self):

        return self.state.value

    # ======================================================
    # Complete Relay Cycle
    # ======================================================

    def process(

        self,

        fault_type,

        confidence

    ):

        self.reset()

        self.update(

            fault_type,

            confidence

        )

        decision = self.trip_decision()

        return {

            "relay_state": self.status(),

            "trip": decision["trip"],

            "fault": decision["fault"]

        }


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    relay = ProtectionRelay()

    samples = [

        ("AG",0.96),

        ("AG",0.94),

        ("Normal",0.30),

        ("AG",0.97),

        ("AG",0.99)

    ]

    for fault, conf in samples:

        result = relay.process(

            fault,

            conf

        )

        print(result)