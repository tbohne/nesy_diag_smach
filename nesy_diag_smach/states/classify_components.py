#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
import os
from typing import List, Dict, Tuple

import numpy as np
import smach
from nesy_ontology import ontology_instance_generator
from oscillogram_classification import cam
from termcolor import colored

from nesy_diag_smach import util
from nesy_diag_smach.config import SESSION_DIR, SUGGESTION_SESSION_FILE, CLASSIFICATION_LOG_FILE
from nesy_diag_smach.data_types.sensor_data import SensorData
from nesy_diag_smach.data_types.state_transition import StateTransition
from nesy_diag_smach.interfaces.data_accessor import DataAccessor
from nesy_diag_smach.interfaces.data_provider import DataProvider
from nesy_diag_smach.interfaces.model_accessor import ModelAccessor


class ClassifyComponents(smach.State):
    """
    State in the SMACH that represents situations in which the suggested physical components are classified:
        - synchronized sensor recordings are performed at the suggested suspect components
        - recorded signals are classified using the trained neural net models, i.e., detecting anomalies
        - manual inspection of suspect components, for which signal diagnosis is not appropriate, is performed
    """

    def __init__(self, model_accessor: ModelAccessor, data_accessor: DataAccessor, data_provider: DataProvider,
                 kg_url: str) -> None:
        """
        Initializes the state.

        :param model_accessor: implementation of the model accessor interface
        :param data_accessor: implementation of the data accessor interface
        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        """
        smach.State.__init__(self,
                             outcomes=['detected_anomalies', 'no_anomaly', 'no_anomaly_no_more_comp'],
                             input_keys=['suggestion_list'],
                             output_keys=['classified_components'])
        self.model_accessor = model_accessor
        self.data_accessor = data_accessor
        self.data_provider = data_provider
        self.instance_gen = ontology_instance_generator.OntologyInstanceGenerator(kg_url=kg_url)

    @staticmethod
    def log_classification_actions(
            classified_components: Dict[str, bool], manually_inspected_components: List[str],
            classification_instances: Dict[str, str]
    ) -> None:
        """
        Logs the classification actions to the session directory.

        :param classified_components: dictionary of classified components + classification results
        :param manually_inspected_components: components that were classified manually by the human
        :param classification_instances: IDs of the classification instances by component name
        """
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "r") as f:
            log_file = json.load(f)
            for k, v in classified_components.items():
                new_data = {
                    k: v,
                    "State": "CLASSIFY_COMPONENTS",
                    "Classification Type": "manual inspection" if k in manually_inspected_components else "signal classification",
                    "Classification ID": classification_instances[k]
                }
                log_file.extend([new_data])
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "w") as f:
            json.dump(log_file, f, indent=4)

    @staticmethod
    def log_state_info() -> None:
        """
        Logs the state information.
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\n############################################")
        print("executing", colored("CLASSIFY_COMPONENTS", "yellow", "on_grey", ["bold"]),
              "state (applying trained model)..")
        print("############################################")

    @staticmethod
    def perform_synchronized_sensor_recordings(
            suggestion_list: Dict[str, Tuple[str, bool]]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Performs synchronized sensor recordings based on the provided suggestion list.

        :param suggestion_list: suspect components suggested for analysis {comp_name: (reason_for, osci_usage)}
        :return: tuple of components to be recorded and components to be verified manually
                 ({comp: reason_for}, {comp: reason_for})
        """
        components_to_be_recorded = {k: v[0] for k, v in suggestion_list.items() if v[1]}
        components_to_be_manually_verified = {k: v[0] for k, v in suggestion_list.items() if not v[1]}
        print("------------------------------------------")
        print("components to be recorded:", components_to_be_recorded)
        print("components to be verified manually:", components_to_be_manually_verified)
        print("------------------------------------------")
        print(colored("\nperform synchronized sensor recordings at:", "green", "on_grey", ["bold"]))
        for comp in components_to_be_recorded.keys():
            print(colored("- " + comp, "green", "on_grey", ["bold"]))
        return components_to_be_recorded, components_to_be_manually_verified

    @staticmethod
    def log_corresponding_error_code(sensor_data: SensorData) -> None:
        """
        Logs the corresponding error code to set the heatmaps for, i.e., reads the error code suggestion.
        Assumption: it is always the latest suggestion.

        :param sensor_data: sensor data (time series signal)
        """
        with open(SESSION_DIR + "/" + SUGGESTION_SESSION_FILE) as f:
            suggestions = json.load(f)
        assert sensor_data.comp_name in list(suggestions.values())[0]
        assert len(suggestions.keys()) == 1
        dtc = list(suggestions.keys())[0]
        print("error code to set heatmap for:", dtc)

    def process_sensor_recordings(
            self, sensor_recordings: List[SensorData], suggestion_list: Dict[str, Tuple[str, bool]],
            anomalous_components: List[str], non_anomalous_components: List[str],
            components_to_be_recorded: Dict[str, str], classification_instances: Dict[str, str]
    ) -> None:
        """
        Iteratively processes the sensor recordings, i.e., classifies each recording and overlays heatmaps.

        :param sensor_recordings: sensor signals to be classified
        :param suggestion_list: suspect components suggested for analysis {comp_name: (reason_for, osci_usage)}
        :param anomalous_components: list to be filled with anomalous components, i.e., detected anomalies
        :param non_anomalous_components: list to be filled with regular components, i.e., no anomalies
        :param components_to_be_recorded: tuple of recorded components
        :param classification_instances: generated classification instances
        """
        for sensor_rec in sensor_recordings:  # iteratively process parallel recorded sensor signals
            sensor_rec_id = self.instance_gen.extend_knowledge_graph_with_oscillogram(sensor_rec.time_series)
            print(colored("\n\nclassifying:" + sensor_rec.comp_name, "green", "on_grey", ["bold"]))
            values = sensor_rec.time_series

            model = self.model_accessor.get_keras_univariate_ts_classification_model_by_component(sensor_rec.comp_name)
            if model is None:
                util.no_trained_model_available(sensor_rec, suggestion_list)
                continue
            (model, model_meta_info) = model  # not only obtain the model here, but also meta info
            values = util.preprocess_time_series_based_on_model_meta_info(model_meta_info, values)
            try:
                util.validate_keras_model(model)
            except ValueError as e:
                util.invalid_model(sensor_rec, suggestion_list, e)
                continue

            net_input = util.construct_net_input(model, values)
            prediction = model.predict(np.array([net_input]))
            num_classes = len(prediction[0])
            # addresses both models with one output neuron and those with several
            anomaly = np.argmax(prediction) == 0 if num_classes > 1 else prediction[0][0] <= 0.5
            pred_value = prediction.max() if num_classes > 1 else prediction[0][0]

            if anomaly:
                util.log_anomaly(pred_value)
                anomalous_components.append(sensor_rec.comp_name)
            else:
                util.log_regular(pred_value)
                non_anomalous_components.append(sensor_rec.comp_name)

            heatmaps = util.gen_heatmaps(net_input, model, prediction)
            res_str = (" [ANOMALY" if anomaly else " [NO ANOMALY") + " - SCORE: " + str(pred_value) + "]"
            self.log_corresponding_error_code(sensor_rec)
            print("heatmap excerpt:", heatmaps["tf-keras-gradcam"][:5])

            # TODO: which heatmap generation method result do we store here? for now, I'll use gradcam
            heatmap_id = self.instance_gen.extend_knowledge_graph_with_heatmap(
                "tf-keras-gradcam", heatmaps["tf-keras-gradcam"].tolist()
            )
            heatmap_img = cam.gen_heatmaps_as_overlay(heatmaps, np.array(values), sensor_rec.comp_name + res_str)
            self.data_provider.provide_heatmaps(heatmap_img, sensor_rec.comp_name + res_str)
            classification_id = self.instance_gen.extend_knowledge_graph_with_sensor_signal_classification(
                anomaly, components_to_be_recorded[sensor_rec.comp_name], sensor_rec.comp_name, pred_value,
                model_meta_info["model_id"], sensor_rec_id, heatmap_id
            )
            classification_instances[sensor_rec.comp_name] = classification_id

    def perform_manual_classifications(
            self, components_to_be_manually_verified: Dict[str, str], classification_instances: Dict[str, str],
            anomalous_components: List[str], non_anomalous_components: List[str]
    ) -> None:
        """
        Classifies the subset of components that are to be classified manually.

        :param components_to_be_manually_verified: components to be verified manually
        :param classification_instances: dictionary of classification instances {comp: classification_ID}
        :param anomalous_components: list of anomalous components (to be extended)
        :param non_anomalous_components: list of regular components (to be extended)
        """
        for comp in components_to_be_manually_verified.keys():
            print(colored("\n\nmanual inspection of component " + comp, "green", "on_grey", ["bold"]))
            anomaly = self.data_accessor.get_manual_judgement_for_component(comp)
            classification_id = self.instance_gen.extend_knowledge_graph_with_manual_inspection(
                anomaly, components_to_be_manually_verified[comp], comp
            )
            classification_instances[comp] = classification_id
            if anomaly:
                anomalous_components.append(comp)
            else:
                non_anomalous_components.append(comp)

    @staticmethod
    def gen_classified_components_dict(
            non_anomalous_components: List[str], anomalous_components: List[str]
    ) -> Dict[str, bool]:
        """
        Generates the dictionary of classified components.

        :param non_anomalous_components: list of regular components
        :param anomalous_components: list of anomalous components
        :return: classified components dict ({comp: anomaly})
        """
        classified_components = {}
        for comp in non_anomalous_components:
            classified_components[comp] = False
        for comp in anomalous_components:
            classified_components[comp] = True
        return classified_components

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'CLASSIFY_COMPONENTS' state.

        :param userdata: input of state
        :return: outcome of the state ("detected_anomalies" | "no_anomaly" | "no_anomaly_no_more_comp")
        """
        self.log_state_info()
        components_to_be_recorded, components_to_be_manually_verified = self.perform_synchronized_sensor_recordings(
            userdata.suggestion_list
        )
        sensor_signals = self.data_accessor.get_signals_by_components(list(components_to_be_recorded.keys()))
        anomalous_components = []
        non_anomalous_components = []
        classification_instances = {}
        self.process_sensor_recordings(
            sensor_signals, userdata.suggestion_list, anomalous_components, non_anomalous_components,
            components_to_be_recorded, classification_instances
        )
        self.perform_manual_classifications(
            components_to_be_manually_verified, classification_instances, anomalous_components, non_anomalous_components
        )
        classified_components = self.gen_classified_components_dict(non_anomalous_components, anomalous_components)
        userdata.classified_components = list(classification_instances.values())
        self.log_classification_actions(
            classified_components, list(components_to_be_manually_verified.keys()), classification_instances
        )
        # there are three options:
        #   1. there's only one recording at a time and thus only one classification
        #   2. there are as many parallel recordings as there are suspect components for the error code
        #   3. there are multiple parallel recordings, but not as many as there are suspect components for the error code
        # TODO: are there remaining suspect components? (atm every component is suggested each case)
        remaining_suspect_components = False

        if len(anomalous_components) == 0 and not remaining_suspect_components:
            self.data_provider.provide_state_transition(StateTransition(
                "CLASSIFY_COMPONENTS", "SELECT_UNUSED_ERROR_CODE", "no_anomaly_no_more_comp"
            ))
            return "no_anomaly_no_more_comp"
        elif len(anomalous_components) == 0 and remaining_suspect_components:
            self.data_provider.provide_state_transition(StateTransition(
                "CLASSIFY_COMPONENTS", "SUGGEST_SUSPECT_COMPONENTS", "no_anomaly"
            ))
            return "no_anomaly"
        elif len(anomalous_components) > 0:
            self.data_provider.provide_state_transition(StateTransition(
                "CLASSIFY_COMPONENTS", "ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS", "detected_anomalies"
            ))
            return "detected_anomalies"
