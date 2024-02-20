#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
from typing import List

from oscillogram_classification import preprocess

from nesy_diag_smach.config import FAULT_CONTEXT_INPUT_FILE
from nesy_diag_smach.config import SIGNAL_SESSION_FILES
from nesy_diag_smach.data_types.fault_context import FaultContext
from nesy_diag_smach.data_types.sensor_data import SensorData
from nesy_diag_smach.interfaces.data_accessor import DataAccessor


class LocalDataAccessor(DataAccessor):
    """
    Implementation of the data accessor interface using local files.
    """

    def __init__(self, verbose: bool = False):
        """
        Initializes the local data accessor.

        :param verbose: sets verbosity of data accessor
        """
        self.verbose = verbose

    def get_fault_context(self) -> FaultContext:
        """
        Retrieves the fault context data required in the diagnostic process.

        :return: fault context data
        """
        val = None
        while val != "":
            val = input("\nlocal interface impl.: processing fault context data..")

        with open(FAULT_CONTEXT_INPUT_FILE, "r") as f:
            problem_instance = json.load(f)

        # only take list of error codes as input, not more
        input_error_codes = list(problem_instance["error_codes"].keys())

        fault_context = FaultContext(input_error_codes, "1234567890ABCDEFGHJKLMNPRSTUVWXYZ")
        print(fault_context)
        return fault_context

    def get_signals_by_components(self, components: List[str]) -> List[SensorData]:
        """
        Retrieves the sensor data for the specified components.

        :param components: components to retrieve sensor data for
        :return: sensor data for each component
        """
        val = None
        while val != "":
            val = input("\nlocal interface impl.: sim human - press 'ENTER' when the recording phase is finished"
                        + " and the signals are generated for " + str(components))
        signals = []

        # for each component we need to check the ground truth of the instance - if it should have an anomaly
        with open(FAULT_CONTEXT_INPUT_FILE, "r") as f:
            problem_instance = json.load(f)

        for comp in components:
            if self.verbose:
                print("GROUND TRUTH ANOMALY:", problem_instance["suspect_components"][comp][0])
            anomaly_suffix = "NEG" if problem_instance["suspect_components"][comp][0] else "POS"
            path = "res/" + SIGNAL_SESSION_FILES + "/" + comp + "/dummy_signal_" + anomaly_suffix + ".csv"
            _, values = preprocess.read_oscilloscope_recording(path)
            signals.append(SensorData(values, comp))
        return signals

    def get_manual_judgement_for_component(self, component: str) -> bool:
        """
        Retrieves a manual judgement by the human for the specified component.

        :param component: component to get manual judgement for
        :return: true -> anomaly, false -> regular
        """
        if self.verbose:
            print("local interface impl.: manual inspection of component:", component)
        val = ""
        while val not in ['0', '1']:
            val = input("\nsim human - press '0' for defective component, i.e., anomaly, and '1' for no defect..")
        return val == "0"

    def get_manual_judgement_for_sensor(self) -> bool:
        """
        Retrieves a manual judgement by the human for the currently considered sensor.

        :return: true -> anomaly, false -> regular
        """
        if self.verbose:
            print("no anomaly identified -- check potential sensor malfunction..")
        val = ""
        while val not in ['0', '1']:
            val = input("\npress '0' for sensor malfunction and '1' for working sensor..")
        return val == "0"
