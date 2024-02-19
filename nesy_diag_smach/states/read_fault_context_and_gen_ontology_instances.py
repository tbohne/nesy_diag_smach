#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
import shutil

import smach
from nesy_diag_ontology import ontology_instance_generator
from termcolor import colored

from nesy_diag_smach.config import SESSION_DIR, FAULT_CONTEXT_FILE, ERROR_CODE_TMP_FILE, CLASSIFICATION_LOG_FILE
from nesy_diag_smach.data_types.fault_context import FaultContext
from nesy_diag_smach.data_types.state_transition import StateTransition
from nesy_diag_smach.interfaces.data_accessor import DataAccessor
from nesy_diag_smach.interfaces.data_provider import DataProvider


class ReadFaultContextAndGenOntologyInstances(smach.State):
    """
    State in the SMACH that represents situations in which the fault context is read.
    Based on the read information, ontology instances are generated, i.e., the diag-subject-specific instance data
    is entered into the knowledge graph.
    """

    def __init__(self, data_accessor: DataAccessor, data_provider: DataProvider, kg_url: str, verbose: bool) -> None:
        """
        Initializes the state.

        :param data_accessor: implementation of the data accessor interface
        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        :param verbose: whether the state machine should log its state, transitions, etc.
        """
        smach.State.__init__(self,
                             outcomes=['processed_fault_context'],
                             input_keys=[''],
                             output_keys=['diag_subject_specific_instance_data'])
        self.data_accessor = data_accessor
        self.data_provider = data_provider
        self.instance_gen = ontology_instance_generator.OntologyInstanceGenerator(kg_url=kg_url)
        self.verbose = verbose

    def log_state_info(self) -> None:
        """
        Logs the state information.
        """
        if self.verbose:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n\n############################################")
            print("executing", colored("READ_FAULT_CONTEXT_AND_GEN_ONTOLOGY_INSTANCES", "yellow", "on_grey", ["bold"]),
                  "state..")
            print("############################################")

    @staticmethod
    def write_fault_context_to_session_file(fault_context: FaultContext) -> None:
        """
        Writes the fault context data to the session directory.

        :param fault_context: fault context data to be stored in session dir
        """
        with open(SESSION_DIR + "/" + FAULT_CONTEXT_FILE, "w") as f:
            json.dump(fault_context.get_json_representation(), f, default=str)

    @staticmethod
    def create_tmp_file_for_unused_error_code_instances(fault_context: FaultContext) -> None:
        """
        Creates a temporary file for unused error code instances.

        :param fault_context: fault context data containing the unused error code instances
        """
        with open(SESSION_DIR + "/" + ERROR_CODE_TMP_FILE, "w") as f:
            error_code_tmp = {'list': fault_context.error_code_list}
            json.dump(error_code_tmp, f, default=str)

    @staticmethod
    def create_session_setup():
        if os.path.exists(SESSION_DIR):
            for filename in os.listdir(SESSION_DIR):
                file_path = os.path.join(SESSION_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"failed to delete {file_path} - {e}")
        else:
            os.makedirs(SESSION_DIR)
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, 'w') as f:
            f.write("[]")

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'READ_FAULT_CONTEXT_AND_GEN_ONTOLOGY_INSTANCES' state.

        :param userdata: input of the state
        :return: outcome of the state ("processed_fault_context")
        """
        self.log_state_info()
        self.create_session_setup()
        fault_context = self.data_accessor.get_fault_context()
        self.write_fault_context_to_session_file(fault_context)

        # extend knowledge graph with read fault context
        self.instance_gen.extend_knowledge_graph_with_diag_subject_data(fault_context.diag_subject_id)

        self.create_tmp_file_for_unused_error_code_instances(fault_context)
        userdata.diag_subject_specific_instance_data = fault_context
        self.data_provider.provide_state_transition(StateTransition(
            "READ_FAULT_CONTEXT_AND_GEN_ONTOLOGY_INSTANCES", "SELECT_UNUSED_ERROR_CODE", "processed_fault_context"
        ))
        return "processed_fault_context"
