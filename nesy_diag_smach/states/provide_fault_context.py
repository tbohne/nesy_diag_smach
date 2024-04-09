#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from datetime import datetime
from typing import Dict, Union, List

import smach
from nesy_diag_ontology import knowledge_graph_query_tool, ontology_instance_generator
from termcolor import colored

from nesy_diag_smach.config import SESSION_DIR, CLASSIFICATION_LOG_FILE, FAULT_CONTEXT_FILE
from nesy_diag_smach.data_types.state_transition import StateTransition
from nesy_diag_smach.interfaces.data_provider import DataProvider


class ProvideFaultContext(smach.State):
    """
    State in the SMACH that represents situations in which only the refuted initial hypothesis as well as
    the context of the diagnostic process is provided due to unmanageable uncertainty.
    """

    def __init__(self, data_provider: DataProvider, kg_url: str, verbose: bool) -> None:
        """
        Initializes the state.

        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        :param verbose: whether the state machine should log its state, transitions, etc.
        """
        smach.State.__init__(self, outcomes=['no_diag'], input_keys=[''], output_keys=['final_output'])
        self.data_provider = data_provider
        self.instance_gen = ontology_instance_generator.OntologyInstanceGenerator(kg_url=kg_url, verbose=verbose)
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=kg_url, verbose=verbose)
        self.verbose = verbose

    def log_state_info(self) -> None:
        """
        Logs the state information.
        """
        if self.verbose:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n\n############################################")
            print("executing", colored("PROVIDE_FAULT_CONTEXT", "yellow", "on_grey", ["bold"]), "state..")
            print("############################################")

    @staticmethod
    def read_metadata() -> Dict[str, Union[int, str]]:
        """
        Reads the metadata from the session directory.

        :return: metadata dictionary
        """
        with open(SESSION_DIR + '/metadata.json', 'r') as f:
            return json.load(f)

    @staticmethod
    def read_fault_context() -> Dict[str, Union[str, List[str]]]:
        """
        Reads the fault context data from the session directory.

        :return: fault context data dictionary
        """
        with open(SESSION_DIR + "/" + FAULT_CONTEXT_FILE, "r") as f:
            return json.load(f)

    @staticmethod
    def read_classification_ids() -> List[str]:
        """
        Reads the classification IDs from the session directory.

        :return: list of classification IDs
        """
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "r") as f:
            log_file = json.load(f)
        return [classification_entry["Classification ID"] for classification_entry in log_file]

    def read_diag_subject_id(self, fault_context) -> str:
        """
        Queries the ID of the diagnosis subject based on the provided fault context.

        :param fault_context: fault context data to query diag subject ID for
        :return: diag subject ID
        """
        return self.qt.query_diag_subject_instance_by_id(fault_context["diag_subject_id"])[0].split("#")[1]

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'PROVIDE_FAULT_CONTEXT' state.

        :param userdata: input of state
        :return: outcome of the state ("no_diag")
        """
        self.log_state_info()
        # TODO: create log file for the failed diagnostic process to improve future diagnosis (missing knowledge etc.)
        self.data_provider.provide_state_transition(StateTransition(
            "PROVIDE_FAULT_CONTEXT", "refuted_hypothesis", "no_diag"
        ))
        # TODO: use metadata in reasonable fashion
        # data = self.read_metadata()
        data = {"diag_date": str(datetime.date)}
        fault_context = self.read_fault_context()
        classification_ids = self.read_classification_ids()
        diag_subject_id = self.read_diag_subject_id(fault_context)

        self.instance_gen.extend_knowledge_graph_with_diag_log(
            data["diag_date"], fault_context["error_code_list"], [], classification_ids, diag_subject_id
        )

        userdata.final_output = "no_diag"
        return "no_diag"
