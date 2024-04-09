#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import logging
import random

import smach
import tensorflow as tf

from nesy_diag_smach.config import KG_URL
from nesy_diag_smach.interfaces.data_accessor import DataAccessor
from nesy_diag_smach.interfaces.data_provider import DataProvider
from nesy_diag_smach.interfaces.model_accessor import ModelAccessor
from nesy_diag_smach.io.local_data_accessor import LocalDataAccessor
from nesy_diag_smach.io.local_data_provider import LocalDataProvider
from nesy_diag_smach.io.local_model_accessor import LocalModelAccessor
from nesy_diag_smach.states.classify_components import ClassifyComponents
from nesy_diag_smach.states.isolate_problem_check_effective_radius import IsolateProblemCheckEffectiveRadius
from nesy_diag_smach.states.no_problem_detected_check_sensor import NoProblemDetectedCheckSensor
from nesy_diag_smach.states.provide_diag_and_show_trace import ProvideDiagAndShowTrace
from nesy_diag_smach.states.provide_fault_context import ProvideFaultContext
from nesy_diag_smach.states.read_fault_context_and_gen_ontology_instances import ReadFaultContextAndGenOntologyInstances
from nesy_diag_smach.states.select_unused_error_code import SelectUnusedErrorCode
from nesy_diag_smach.states.suggest_suspect_components import SuggestSuspectComponents
from nesy_diag_smach.util import log_info, log_debug, log_warn, log_err


class NeuroSymbolicDiagnosisStateMachine(smach.StateMachine):
    """
    Neuro-symbolic state machine guiding the entire diagnosis process.
    """

    def __init__(
            self, data_accessor: DataAccessor, model_accessor: ModelAccessor, data_provider: DataProvider,
            kg_url: str = KG_URL, verbose: bool = True, sim_models: bool = False, seed: int = 0
    ) -> None:
        """
        Initializes the neuro-symbolic diagnosis state machine.

        :param data_accessor: implementation of the data accessor interface
        :param model_accessor: implementation of the model accessor interface
        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        :param verbose: whether the state machine should log its state, transitions, etc.
        :param sim_models: whether the classification models should be simulated
        :param seed: seed for random processes
        """
        super(NeuroSymbolicDiagnosisStateMachine, self).__init__(
            outcomes=['diag', 'refuted_hypothesis'],
            input_keys=[],
            output_keys=['final_output']
        )
        self.data_accessor = data_accessor
        self.model_accessor = model_accessor
        self.data_provider = data_provider
        self.userdata.sm_input = []
        self.kg_url = kg_url
        self.verbose = verbose
        self.sim_models = sim_models

        random.seed(seed)
        _, num_of_sim_models = self.model_accessor.get_sim_univariate_ts_classification_model_by_component("C0")
        comp_indices = [i for i in range(num_of_sim_models)]
        self.random_value_dict = {
            # throw a dice based on model probability - random val from [0, 1]
            "C" + str(i): random.random() for i in comp_indices
        }

        with self:
            self.add('READ_FAULT_CONTEXT_AND_GEN_ONTOLOGY_INSTANCES',
                     ReadFaultContextAndGenOntologyInstances(
                         self.data_accessor, self.data_provider, self.kg_url, self.verbose
                     ),
                     transitions={'processed_fault_context': 'SELECT_UNUSED_ERROR_CODE'},
                     remapping={'input': 'sm_input', 'user_data': 'sm_input'})

            self.add('SELECT_UNUSED_ERROR_CODE', SelectUnusedErrorCode(self.data_provider, self.verbose),
                     transitions={'selected_best_instance': 'SUGGEST_SUSPECT_COMPONENTS',
                                  'no_instance': 'NO_PROBLEM_DETECTED_CHECK_SENSOR',
                                  'no_instance_prev_diag': 'PROVIDE_DIAG_AND_SHOW_TRACE'},
                     remapping={'selected_instance': 'sm_input', 'fault_paths': 'sm_input'})

            self.add('ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS',
                     IsolateProblemCheckEffectiveRadius(
                         self.data_accessor, self.model_accessor, self.data_provider, self.kg_url, self.verbose,
                         self.sim_models, self.random_value_dict
                     ),
                     transitions={'isolated_problem': 'PROVIDE_DIAG_AND_SHOW_TRACE',
                                  'isolated_problem_remaining_error_codes': 'SELECT_UNUSED_ERROR_CODE'},
                     remapping={'classified_components': 'sm_input', 'fault_paths': 'sm_input'})

            self.add('NO_PROBLEM_DETECTED_CHECK_SENSOR',
                     NoProblemDetectedCheckSensor(self.data_accessor, self.data_provider, self.verbose),
                     transitions={'sensor_works': 'PROVIDE_FAULT_CONTEXT',
                                  'sensor_defective': 'PROVIDE_DIAG_AND_SHOW_TRACE'},
                     remapping={})

            self.add('PROVIDE_FAULT_CONTEXT',
                     ProvideFaultContext(self.data_provider, self.kg_url, self.verbose),
                     transitions={'no_diag': 'refuted_hypothesis'},
                     remapping={'final_output': 'final_output'})

            self.add('PROVIDE_DIAG_AND_SHOW_TRACE',
                     ProvideDiagAndShowTrace(self.data_provider, self.kg_url, self.verbose),
                     transitions={'uploaded_diag': 'diag'},
                     remapping={'diagnosis': 'sm_input', 'final_output': 'final_output'})

            self.add('CLASSIFY_COMPONENTS',
                     ClassifyComponents(
                         self.model_accessor, self.data_accessor, self.data_provider, self.kg_url, self.verbose,
                         self.sim_models, self.random_value_dict
                     ),
                     transitions={'no_anomaly_no_more_comp': 'SELECT_UNUSED_ERROR_CODE',
                                  'no_anomaly': 'SUGGEST_SUSPECT_COMPONENTS',
                                  'detected_anomalies': 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS'},
                     remapping={'suggestion_list': 'sm_input', 'classified_components': 'sm_input'})

            self.add('SUGGEST_SUSPECT_COMPONENTS',
                     SuggestSuspectComponents(self.data_provider, self.kg_url, self.verbose),
                     transitions={'provided_suggestions': 'CLASSIFY_COMPONENTS'},
                     remapping={'selected_instance': 'sm_input', 'suggestion_list': 'sm_input'})


if __name__ == '__main__':
    smach.set_loggers(log_info, log_debug, log_warn, log_err)  # set custom logging functions

    # init local implementations of I/O interfaces
    data_acc = LocalDataAccessor()
    model_acc = LocalModelAccessor()
    data_prov = LocalDataProvider()

    sm = NeuroSymbolicDiagnosisStateMachine(data_acc, model_acc, data_prov)
    tf.get_logger().setLevel(logging.ERROR)
    sm.execute()
    print("final output of smach execution (fault path(s)):", sm.userdata.final_output)
