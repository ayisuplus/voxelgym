import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Shared helpers for the physics truth-table suites (were duplicated
# byte-for-byte in test_m3.py and test_m35.py).
IDLE = (0, 0, 0, 0, 4, 0, 0, 0, 0, 0)


def cid(cell):
    return cell & 0xFFF


def state(cell):
    return cell >> 12


def run(w, ticks):
    for _ in range(ticks):
        w.step(IDLE)
