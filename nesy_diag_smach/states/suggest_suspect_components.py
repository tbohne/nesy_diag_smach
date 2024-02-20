#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from typing import List, Dict, Tuple

import smach
from nesy_diag_ontology import knowledge_graph_query_tool
from termcolor import colored

from nesy_diag_smach.config import SESSION_DIR, SUS_COMP_TMP_FILE, SUGGESTION_SESSION_FILE
from nesy_diag_smach.data_types.state_transition import StateTransition
from nesy_diag_smach.interfaces.data_provider import DataProvider


class SuggestSuspectComponents(smach.State):
    """
    State in the SMACH that represents situations in which suspect components (physical components) in the
    diag subject are suggested to be investigated based on the available information.
    """

    def __init__(self, data_provider: DataProvider, kg_url: str, verbose: bool) -> None:
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        :param verbose: whether the state machine should log its state, transitions, etc.
        """
        smach.State.__init__(self,
                             outcomes=['provided_suggestions'],
                             input_keys=['selected_instance'],
                             output_keys=['suggestion_list'])
        self.data_provider = data_provider
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=kg_url, verbose=verbose)
        self.verbose = verbose

    def log_state_info(self) -> None:
        """
        Logs the state information.
        """
        if self.verbose:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n\n############################################")
            print("executing", colored("SUGGEST_SUSPECT_COMPONENTS", "yellow", "on_grey", ["bold"]), "state..")
            print("############################################\n")

    @staticmethod
    def write_components_to_file(suspect_components: List[str]) -> None:
        """
        Writes the suspect components to a session file.

        :param suspect_components: components to be stored in session file
        """
        with open(SESSION_DIR + "/" + SUS_COMP_TMP_FILE, 'w') as f:
            json.dump(suspect_components, f, default=str)

    @staticmethod
    def read_components_from_file() -> List[str]:
        """
        Reads the remaining suspect components from file.

        :return: list of remaining suspect components
        """
        with open(SESSION_DIR + "/" + SUS_COMP_TMP_FILE) as f:
            return json.load(f)

    @staticmethod
    def write_suggestions_to_session_file(selected_instance: str, suspect_components: List[str]) -> None:
        """
        Writes the suggestions to a session file - always the latest ones.

        :param selected_instance: selected error code instance
        :param suspect_components: list of suggested suspect components
        """
        suggestion = {selected_instance: str(suspect_components)}
        with open(SESSION_DIR + "/" + SUGGESTION_SESSION_FILE, 'w') as f:
            json.dump(suggestion, f, default=str)

    def determine_sensor_usage(self, suspect_components: List[str]) -> List[bool]:
        """
        Decides whether a sensor is required / feasible for each component.

        :param suspect_components: components to determine sensor usage for
        :return: booleans representing sensor usage for each component
        """
        sensor_usage = []
        for comp in suspect_components:
            # TODO: for now, we expect that all components can be diagnosed based on a sensor signal
            use = True  # self.qt.query_sensor_usage_by_suspect_component(comp)[0]
            if self.verbose:
                print("comp:", comp, "// use sensor:", use)
            sensor_usage.append(use)
        return sensor_usage

    def gen_suggestions(self, selected_instance: str, suspect_components: List[str], sensor_usage: List[bool]) \
            -> Dict[str, Tuple[str, bool]]:
        """
        Generates the suggestion dictionary: {comp: (reason_for, anomaly)}.

        :param selected_instance: selected error code instance
        :param suspect_components: suggested suspect components
        :param sensor_usage: sensor usage for the suggested components
        :return: suggestion dictionary
        """
        return {
            comp: (
                self.qt.query_diag_association_instance_by_error_code_and_sus_comp(
                    selected_instance, comp
                )[0].split("#")[1], sensor
            ) for comp, sensor in zip(suspect_components, sensor_usage)
        }

    @staticmethod
    def update_session_file(suspect_components: List[str], suggestions: Dict[str, Tuple[str, bool]]) -> None:
        """
        Everything that is used here should be removed from the tmp file.

        :param suspect_components: suggested components
        :param suggestions: suggestion dictionary
        """
        with open(SESSION_DIR + "/" + SUS_COMP_TMP_FILE, 'w') as f:
            json.dump([c for c in suspect_components if c not in suggestions.keys()], f, default=str)

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'SUGGEST_SUSPECT_COMPONENTS' state.

        :param userdata: input of state
        :return: outcome of the state ("provided_suggestions")
        """
        self.log_state_info()
        # should not be queried over and over again - just once for a session
        # -> then suggest as many as possible per execution of the state (write to session files)
        if not os.path.exists(SESSION_DIR + "/" + SUS_COMP_TMP_FILE) or len(self.read_components_from_file()) == 0:
            suspect_components = self.qt.query_suspect_components_by_error_code(userdata.selected_instance)
            ordered_sus_comp = {  # sort suspect components
                int(self.qt.query_priority_id_by_error_code_and_sus_comp(userdata.selected_instance, comp, False)[0]):
                    comp for comp in suspect_components
            }
            suspect_components = [ordered_sus_comp[i] for i in range(len(suspect_components))]
            self.write_components_to_file(suspect_components)
        else:
            suspect_components = self.read_components_from_file()
        if self.verbose:
            print(colored("SUSPECT COMPONENTS: " + str(suspect_components) + "\n", "green", "on_grey", ["bold"]))
        self.write_suggestions_to_session_file(userdata.selected_instance, suspect_components)
        sensor_usage = self.determine_sensor_usage(suspect_components)
        suggestions = self.gen_suggestions(userdata.selected_instance, suspect_components, sensor_usage)
        userdata.suggestion_list = suggestions
        self.update_session_file(suspect_components, suggestions)

        if self.verbose:
            if True in sensor_usage:
                print("\n--> there is at least one suspect component that can be diagnosed using a sensor signal..")
            else:
                print("\n--> none of the identified suspect components can be diagnosed with a sensor signal..")

        self.data_provider.provide_state_transition(StateTransition(
            "SUGGEST_SUSPECT_COMPONENTS", "CLASSIFY_COMPONENTS", "provided_suggestions"
        ))
        return "provided_suggestions"
