import numpy as np
import time


def radian2degree(x):
    return x * 180 / np.pi


def degree2radian(x):
    return x * np.pi / 180


class Timer:
    """
    Time counter of operations
    """

    start_perf = 0

    def __init__(self):
        self.start_perf = time.perf_counter()

    def end(self):
        return time.perf_counter() - self.start_perf

    def begin(self):
        self.start_perf = time.perf_counter()
