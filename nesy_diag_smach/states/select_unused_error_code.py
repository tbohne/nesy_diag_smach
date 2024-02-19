#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from typing import List

import smach
from termcolor import colored

from nesy_diag_smach import util
from nesy_diag_smach.config import SESSION_DIR, ERROR_CODE_TMP_FILE
from nesy_diag_smach.data_types.state_transition import StateTransition
from nesy_diag_smach.interfaces.data_provider import DataProvider


class SelectUnusedErrorCode(smach.State):
    """
    State in the SMACH that represents situations in which a best-suited, unused error code instance is
    selected for further processing.
    """

    def __init__(self, data_provider: DataProvider, verbose: bool) -> None:
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        :param verbose: whether the state machine should log its state, transitions, etc.
        """
        smach.State.__init__(self,
                             outcomes=['selected_best_instance', 'no_instance'],
                             input_keys=[''],
                             output_keys=['selected_instance'])
        self.data_provider = data_provider
        self.verbose = verbose

    @staticmethod
    def remove_error_code_instance_from_tmp_file(remaining_instances: List[str]) -> None:
        """
        Updates the list of unused error code instances in the corresponding tmp file.

        :param remaining_instances: updated list to save in tmp file
        """
        with open(SESSION_DIR + "/" + ERROR_CODE_TMP_FILE, "w") as f:
            json.dump({'list': remaining_instances}, f, default=str)

    def log_state_info(self) -> None:
        """
        Logs the state information.
        """
        if self.verbose:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n\n############################################")
            print("executing", colored("SELECT_UNUSED_ERROR_CODE", "yellow", "on_grey", ["bold"]), "state..")
            print("############################################")

    def remaining_error_codes(self, error_code_list: List[str], selected_error_code: str) -> None:
        """
        Handles cases with remaining error codes.

        :param error_code_list: list of error codes
        :param selected_error_code: selected error code
        """
        error_code_list.remove(selected_error_code)
        self.remove_error_code_instance_from_tmp_file(error_code_list)
        if self.verbose:
            print(colored("selected error code instance: " + selected_error_code, "green", "on_grey", ["bold"]))
        self.data_provider.provide_state_transition(StateTransition(
            "SELECT_UNUSED_ERROR_CODE", "SUGGEST_SUSPECT_COMPONENTS", "selected_best_instance"
        ))

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'SELECT_UNUSED_ERROR_CODE' state.

        :param userdata: input of state
        :return: outcome of the state ("selected_best_instance" | "no_instance")
        """
        self.log_state_info()
        error_code_list = util.load_error_code_instances()

        # no error code instance provided
        if len(error_code_list) == 0:
            return "no_instance"

        # TODO: select best remaining error code instance based on some criteria
        selected_error_code = error_code_list[0]
        userdata.selected_instance = selected_error_code
        self.remaining_error_codes(error_code_list, selected_error_code)
        return "selected_best_instance"
