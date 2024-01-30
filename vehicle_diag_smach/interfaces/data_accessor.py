#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from abc import ABC, abstractmethod
from typing import List

from vehicle_diag_smach.data_types.fault_context import FaultContext
from vehicle_diag_smach.data_types.sensor_data import SensorData


class DataAccessor(ABC):
    """
    Interface that defines the state machine's access to signals and diagnosis-relevant case data.
    """

    @abstractmethod
    def get_fault_context(self) -> FaultContext:
        """
        Retrieves the fault context data required in the diagnostic process.

        :return: fault context data
        """
        pass

    @abstractmethod
    def get_signals_by_components(self, components: List[str]) -> List[SensorData]:
        """
        Retrieves the sensor data data for the specified components.

        :param components: components to retrieve sensor data for
        :return: sensor data data for each component
        """
        pass

    @abstractmethod
    def get_manual_judgement_for_component(self, component: str) -> bool:
        """
        Retrieves a manual judgement by the human for the specified component.

        :param component: component to get manual judgement for
        :return: true -> anomaly, false -> regular
        """
        pass

    @abstractmethod
    def get_manual_judgement_for_sensor(self) -> bool:
        """
        Retrieves a manual judgement by the human for the currently considered sensor.

        :return: true -> anomaly, false -> regular
        """
        pass
