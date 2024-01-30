#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from typing import List


class SensorData:
    """
    Represents sensor data communicated to the state machine.
    """

    def __init__(self, time_series: List[float], comp_name: str) -> None:
        """
        Inits the sensor data.

        The list of values should never be empty, at least two data points are expected.

        :param time_series: recorded signal (values over time)
        :param comp_name: name of the component the signal belongs to
        """
        assert len(time_series) >= 2
        self.time_series = time_series
        self.comp_name = comp_name
