#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import io
import json
import os
import random
from collections import defaultdict
from typing import Union, List, Tuple, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import smach
from PIL import Image
from matplotlib.lines import Line2D
from nesy_diag_ontology import knowledge_graph_query_tool
from nesy_diag_ontology import ontology_instance_generator
from oscillogram_classification import cam
from tensorflow import keras
from termcolor import colored

from nesy_diag_smach import util
from nesy_diag_smach.config import SESSION_DIR, SUGGESTION_SESSION_FILE, SIGNAL_SESSION_FILES, \
    CLASSIFICATION_LOG_FILE, FAULT_PATH_TMP_FILE, SIM_CLASSIFICATION_LOG_FILE, ANOMALY_GRAPH_TMP_FILE
from nesy_diag_smach.data_types.state_transition import StateTransition
from nesy_diag_smach.interfaces.data_accessor import DataAccessor
from nesy_diag_smach.interfaces.data_provider import DataProvider
from nesy_diag_smach.interfaces.model_accessor import ModelAccessor


class IsolateProblemCheckEffectiveRadius(smach.State):
    """
    State in the SMACH that represents situations in which one or more anomalies have been detected, and the
    task is to isolate the defective components based on their effective radius (structural knowledge).
    """

    def __init__(
            self, data_accessor: DataAccessor, model_accessor: ModelAccessor, data_provider: DataProvider, kg_url: str,
            verbose: bool, sim_models: bool, seed: int
    ) -> None:
        """
        Initializes the state.

        :param data_accessor: implementation of the data accessor interface
        :param model_accessor: implementation of the model accessor interface
        :param data_provider: implementation of the data provider interface
        :param kg_url: URL of the knowledge graph guiding the diagnosis
        :param verbose: whether the state machine should log its state, transitions, etc.
        :param sim_models: whether the classification models should be simulated
        :param seed: seed for random processes
        """
        smach.State.__init__(self,
                             outcomes=['isolated_problem', 'isolated_problem_remaining_error_codes'],
                             input_keys=['classified_components'],
                             output_keys=['fault_paths'])
        self.qt = knowledge_graph_query_tool.KnowledgeGraphQueryTool(kg_url=kg_url, verbose=verbose)
        self.instance_gen = ontology_instance_generator.OntologyInstanceGenerator(kg_url=kg_url, verbose=verbose)
        self.data_accessor = data_accessor
        self.model_accessor = model_accessor
        self.data_provider = data_provider
        self.verbose = verbose
        self.sim_models = sim_models
        random.seed(seed)

    @staticmethod
    def create_session_data_dir() -> None:
        """
        Creates the session data directory.
        """
        signal_iso_session_dir = SESSION_DIR + "/" + SIGNAL_SESSION_FILES + "/"
        if not os.path.exists(signal_iso_session_dir):
            os.makedirs(signal_iso_session_dir)

    def get_model_and_metadata(self, affecting_comp: str) -> Tuple[keras.models.Model, dict]:
        """
        Retrieves the trained model and the corresponding metadata.

        :param affecting_comp: component to retrieve trained model for
        :return: tuple of trained model and corresponding metadata
        """
        model = self.model_accessor.get_keras_univariate_ts_classification_model_by_component(affecting_comp)
        if model is None:
            pass  # TODO: handle model is None cases
        (model, model_meta_info) = model
        try:
            util.validate_keras_model(model)
        except ValueError as e:
            print("invalid model for the signal (component) to be classified:", affecting_comp)
            print("error:", e, "\nadding it to the list of components to be verified manually..")
            # TODO: actually handle the case
        return model, model_meta_info

    def provide_heatmaps(
            self, affecting_comp: str, res_str: str, heatmaps: Dict[str, np.ndarray], values: List[float]
    ) -> None:
        """
        Provides the generated heatmaps via the data provider.

        :param affecting_comp: component to classify signal for
        :param res_str: result string (classification result + score)
        :param heatmaps: heatmaps to be provided
        :param values: classified time series values
        """
        title = affecting_comp + "_" + res_str
        # TODO: should be real time values at some point
        time_vals = [i for i in range(len(values))]
        heatmap_img = cam.gen_heatmaps_as_overlay(heatmaps, np.array(values), title, time_vals)
        self.data_provider.provide_heatmaps(heatmap_img, title)

    def classify_component(
            self, affecting_comp: str, error_code: str, classification_reason: str,
            sim_model_data: Dict[str, Tuple[bool, bool, float, float]]
    ) -> Union[Tuple[bool, str], None]:
        """
        Classifies the sensor signal for the specified component.

        :param affecting_comp: component to classify sensor signal for
        :param error_code: error code the original component suggestion was based on
        :param classification_reason: reason for the classification (ID of another classification)
        :param sim_model_data: dictionary mapping comp name to tuple of ground truth anomaly, anomaly, pred, model acc
        :return: tuple of whether an anomaly has been detected and the corresponding classification ID
        """
        self.create_session_data_dir()
        # in this state, there is only one component to be classified, but there could be several
        sensor_signals = self.data_accessor.get_signals_by_components([affecting_comp])
        assert len(sensor_signals) == 1
        values = sensor_signals[0].time_series
        signal_id = self.instance_gen.extend_knowledge_graph_with_time_series(values)

        if self.sim_models:
            sim_accuracies = self.model_accessor.get_sim_univariate_ts_classification_model_by_component(affecting_comp)
            model_acc = float(sim_accuracies[0])
            ground_truth_anomaly = True if sim_accuracies[1] == "True" else False
            # throw a dice based on model probability
            pred_val = random.random()  # random val from [0, 1]
            # if ground truth anomaly, we find it in model_acc % of the cases, if not, we have an FP in model_acc %
            anomaly = pred_val < model_acc if ground_truth_anomaly else pred_val > model_acc

            if self.verbose:
                print("sim classification..")
                if ground_truth_anomaly != anomaly:
                    print("ground truth anomaly:", ground_truth_anomaly)
                    print("predicted anomaly:", anomaly, "(", pred_val, ")")
                    print("model acc.:", model_acc)
            model_id = "sim_model"
            heatmap_id = "sim_model_no_heatmap"
            sim_model_data[affecting_comp] = (ground_truth_anomaly, anomaly, pred_val, model_acc)
        else:
            model, model_meta_info = self.get_model_and_metadata(affecting_comp)
            values = util.preprocess_time_series_based_on_model_meta_info(model_meta_info, values, verbose=False)
            net_input = util.construct_net_input(model, values)
            prediction = model.predict(np.array([net_input]), verbose=self.verbose)
            num_classes = len(prediction[0])
            pred_val = prediction[0][0]
            # addresses both models with one output neuron and those with several
            anomaly = np.argmax(prediction) == 0 if num_classes > 1 else pred_val <= 0.5
            heatmaps = util.gen_heatmaps(net_input, model, prediction)
            res_str = (" [ANOMALY" if anomaly else " [NO ANOMALY") + " - SCORE: " + str(pred_val) + "]"
            if self.verbose:
                print("error code to set heatmap for:", error_code, "\nheatmap excerpt:",
                      heatmaps["tf-keras-gradcam"][:5])
            # TODO: which heatmap generation method result do we store here? for now, I'll use gradcam
            heatmap_id = self.instance_gen.extend_knowledge_graph_with_heatmap(
                "tf-keras-gradcam", heatmaps["tf-keras-gradcam"].tolist()
            )
            self.provide_heatmaps(affecting_comp, res_str, heatmaps, values)
            model_id = model_meta_info['model_id']

        classification_id = self.instance_gen.extend_knowledge_graph_with_signal_classification(
            anomaly, classification_reason, affecting_comp, pred_val, model_id,
            signal_id, heatmap_id
        )
        if self.verbose:
            if anomaly:
                util.log_anomaly(pred_val)
            else:
                util.log_regular(pred_val)
        return anomaly, classification_id

    def construct_complete_graph(
            self, graph: Dict[str, List[str]], components_to_process: List[str]
    ) -> Dict[str, List[str]]:
        """
        Recursive function that constructs the complete causal graph for the specified components.

        :param graph: partial graph to be extended
        :param components_to_process: components yet to be processed
        :return: constructed causal graph
        """
        if len(components_to_process) == 0:
            return graph
        comp = components_to_process.pop(0)
        if comp not in graph.keys():
            affecting_comp = self.qt.query_affected_by_relations_by_suspect_component(comp, False)
            components_to_process += affecting_comp
            graph[comp] = affecting_comp
        return self.construct_complete_graph(graph, components_to_process)

    @staticmethod
    def create_legend_line(color: str, **kwargs) -> Line2D:
        """
        Creates the edge representations for the plot legend.

        :param color: color for legend line
        :return: generated line representation
        """
        return Line2D([0, 1], [0, 1], color=color, **kwargs)

    @staticmethod
    def compute_causal_links(
            to_relations: List[str], key: str, anomalous_paths: Dict[str, List[List[str]]], from_relations: List[str]
    ) -> List[int]:
        """
        Computes the causal links in the subgraph of cause-effect relationships.

        :param to_relations: 'to relations' of the considered subgraph
        :param key: considered component
        :param anomalous_paths: (branching) paths to the root cause
        :param from_relations: 'from relations' of the considered subgraph
        :return: causal links in the subgraph
        """
        causal_links = []
        for i in range(len(to_relations)):
            if key in anomalous_paths.keys():
                for j in range(len(anomalous_paths[key])):
                    for k in range(len(anomalous_paths[key][j]) - 1):
                        # causal link check
                        if (anomalous_paths[key][j][k] == from_relations[i]
                                and anomalous_paths[key][j][k + 1] == to_relations[i]):
                            causal_links.append(i)
                            break
        return causal_links

    @staticmethod
    def set_edge_properties(
            causal_links: List[int], to_relations: List[str], from_relations: List[str],
            explicitly_considered_links: Dict[str, List[str]]
    ) -> Tuple[List[str], List[int]]:
        """
        Sets the edge properties for the causal graph, i.e., sets edge colors and widths.

        :param causal_links: causal links in the subgraph
        :param to_relations: 'to relations' of the considered subgraph
        :param from_relations: 'from relations' of the considered subgraph
        :param explicitly_considered_links: links that have been verified explicitly
        :return: tuple of edge colors and widths
        """
        colors = ['g' if i not in causal_links else 'r' for i in range(len(to_relations))]
        for i in range(len(from_relations)):
            # if the from-to relation is not part of the actually considered links, it should be black
            if from_relations[i] not in explicitly_considered_links.keys() or to_relations[i] not in \
                    explicitly_considered_links[from_relations[i]]:
                colors[i] = 'black'
        widths = [8 if i not in causal_links else 10 for i in range(len(to_relations))]
        return colors, widths

    def gen_causal_graph_visualizations(
            self, anomalous_paths: Dict[str, List[List[str]]], complete_graphs: Dict[str, Dict[str, List[str]]],
            explicitly_considered_links: Dict[str, List[str]]
    ) -> List[Image.Image]:
        """
        Visualizes the causal graphs along with the actual paths to the root cause.

        :param anomalous_paths: the paths to the root cause
        :param complete_graphs: the causal graphs
        :param explicitly_considered_links: links that have been verified explicitly
        :return: causal graph visualizations
        """
        visualizations = []
        for key in anomalous_paths.keys():
            if self.verbose:
                print("isolation results, i.e., causal path:\n", key, ":", anomalous_paths[key])
        for key in complete_graphs.keys():
            if self.verbose:
                print("visualizing graph for component:", key, "\n")
            plt.figure(figsize=(25, 18))
            plt.title("Causal Graph (Network of Effective Connections) for " + key, fontsize=24, fontweight='bold')
            from_relations = [k for k in complete_graphs[key].keys() for _ in range(len(complete_graphs[key][k]))]
            to_relations = [complete_graphs[key][k] for k in complete_graphs[key].keys()]
            to_relations = [item for lst in to_relations for item in lst]
            causal_links = self.compute_causal_links(to_relations, key, anomalous_paths, from_relations)
            colors, widths = self.set_edge_properties(
                causal_links, to_relations, from_relations, explicitly_considered_links
            )
            df = pd.DataFrame({'from': from_relations, 'to': to_relations})
            g = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
            pos = nx.spring_layout(g, scale=0.3, seed=5)
            nx.draw(
                g, pos=pos, with_labels=True, node_size=30000, font_size=10, alpha=0.75, arrows=True, edge_color=colors,
                width=widths
            )
            legend_lines = [self.create_legend_line(clr, lw=5) for clr in ['r', 'g', 'black']]
            labels = ["fault path", "non-anomalous links", "disregarded"]

            # initial preview does not require a legend
            if len(anomalous_paths.keys()) > 0 and len(explicitly_considered_links.keys()) > 0:
                plt.legend(legend_lines, labels, fontsize=20, loc='lower right')

            buf = io.BytesIO()  # create bytes object and save matplotlib fig into it
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            image = Image.open(buf)  # create PIL image object
            visualizations.append(image)
        return visualizations

    @staticmethod
    def log_classification_action(comp: str, anomaly: bool, use_sensor_rec: bool, classification_id: str) -> None:
        """
        Logs the classification actions to the session directory.

        :param comp: classified component
        :param anomaly: whether an anomaly was identified
        :param use_sensor_rec: whether a sensor recording was used for the classification
        :param classification_id: ID of the corresponding classification instance
        """
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "r") as f:
            log_file = json.load(f)
            log_file.extend([{
                comp: anomaly,
                "State": "ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS",
                "Classification Type": "manual inspection" if not use_sensor_rec else "signal classification",
                "Classification ID": classification_id
            }])
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "w") as f:
            json.dump(log_file, f, indent=4)

    @staticmethod
    def log_sim_classification_action(
            comp: str, anomaly: bool, use_sensor_rec: bool, classification_id: str,
            sim_model_data: Dict[str, Tuple[bool, bool, float, float]]
    ) -> None:
        """
        Logs the simulated classification actions to the session directory.

        :param comp: classified component
        :param anomaly: whether an anomaly was identified
        :param use_sensor_rec: whether a sensor recording was used for the classification
        :param classification_id: ID of the corresponding classification instance
        :param sim_model_data: dictionary mapping comp name to tuple of ground truth anomaly, anomaly, pred, model acc
        """
        with open(SESSION_DIR + "/" + SIM_CLASSIFICATION_LOG_FILE, "r") as f:
            log_file = json.load(f)
            log_file.extend([{
                comp: anomaly,
                "Model Accuracy": sim_model_data[comp][3],
                "Predicted Value": sim_model_data[comp][2],
                "Ground Truth Anomaly": sim_model_data[comp][0],
                "State": "ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS",
                "Classification Type": "manual inspection" if not use_sensor_rec else "signal classification",
                "Classification ID": classification_id
            }])
        with open(SESSION_DIR + "/" + SIM_CLASSIFICATION_LOG_FILE, "w") as f:
            json.dump(log_file, f, indent=4)

    def log_state_info(self) -> None:
        """
        Logs the state information.
        """
        if self.verbose:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n\n############################################")
            print("executing", colored("ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS", "yellow", "on_grey", ["bold"]),
                  "state..")
            print("############################################\n")

    def retrieve_already_checked_components(self, classified_components: List[str]) -> Dict[str, Tuple[bool, str]]:
        """
        Retrieves the already checked components together with the corresponding results:
            {comp: (prediction, classification_id)}
        A prediction of "true" stands for a detected anomaly.

        :param classified_components: list of classified components (IDs)
        :return: dictionary of already checked components: {comp: (prediction, classification_id)}
        """
        already_checked_components = {}
        for classification_id in classified_components:
            sus_comp_resp = self.qt.query_suspect_component_by_classification(classification_id)
            assert len(sus_comp_resp) == 1
            comp_id = sus_comp_resp[0].split("#")[1]
            comp_name = self.qt.query_suspect_component_name_by_id(comp_id)[0]
            # the prediction is retrieved as a string, not boolean, thus the check
            pred = self.qt.query_prediction_by_classification(classification_id)[0].lower() == "true"
            already_checked_components[comp_name] = (pred, classification_id)
        return already_checked_components

    @staticmethod
    def read_error_code_suggestion(anomalous_comp: str) -> str:
        """
        Reads the error code the component suggestion was based on - assumption: it is always the latest suggestion.

        :param anomalous_comp: component to read the error code the suggestion was based on for
        :return: read error code
        """
        with open(SESSION_DIR + "/" + SUGGESTION_SESSION_FILE) as f:
            suggestions = json.load(f)
        assert anomalous_comp in list(suggestions.values())[0]
        assert len(suggestions.keys()) == 1
        return list(suggestions.keys())[0]

    def visualize_initial_graph(
            self, anomalous_paths: Dict[str, List[List[str]]], complete_graphs: Dict[str, Dict[str, List[str]]],
            explicitly_considered_links: Dict[str, List[str]]
    ) -> None:
        """
        Visualizes the initial graph (without highlighted edges / pre isolation).

        :param anomalous_paths: the paths to the root cause
        :param complete_graphs: the causal graphs
        :param explicitly_considered_links: links that have been verified explicitly
        """
        visualizations = self.gen_causal_graph_visualizations(
            anomalous_paths, complete_graphs, explicitly_considered_links
        )
        self.data_provider.provide_causal_graph_visualizations(visualizations)

    def retrieve_sus_comp(self, class_id: str) -> str:
        """
        Retrieves the anomalous suspect component specified by the provided classification ID.

        :param class_id: classification ID to retrieve component for
        :return: suspect component for specified classification ID
        """
        sus_comp_resp = self.qt.query_suspect_component_by_classification(class_id)
        assert len(sus_comp_resp) == 1
        comp_id = sus_comp_resp[0].split("#")[1]
        return self.qt.query_suspect_component_name_by_id(comp_id)[0]

    def handle_anomaly(
            self, checked_comp: str, unisolated_anomalous_components: List[str],
            explicitly_considered_links: Dict[str, List[str]], classified_components: Dict[str, Tuple[bool, str]],
            prev_classified_components: Dict[str, Tuple[bool, str]]
    ) -> None:
        """
        Handles anomaly cases, i.e., extends unisolated anomalous components and explicitly considered links.

        :param checked_comp: checked component (found anomaly)
        :param unisolated_anomalous_components: list of unisolated anomalous components to be extended
        :param explicitly_considered_links: list of explicitly considered links to be extended
        :param classified_components: dict of already checked components (in classification state)
        :param prev_classified_components: dict of already checked components (in previous iterations)
        """
        affecting_comps = self.qt.query_affected_by_relations_by_suspect_component(checked_comp)
        if self.verbose:
            print("component potentially affected by:", affecting_comps)
        already_considered_anomaly_links = [comp for comp in explicitly_considered_links[checked_comp] if (
                comp in classified_components and classified_components[comp][0]
                or comp in prev_classified_components and prev_classified_components[comp][0]
        )]
        not_yet_visited = [comp for comp in affecting_comps if comp not in already_considered_anomaly_links]
        unisolated_anomalous_components += not_yet_visited
        explicitly_considered_links[checked_comp] += affecting_comps.copy()

    def work_through_unisolated_components(
            self, unisolated_comps: List[str], explicitly_considered_links: Dict[str, List[str]],
            classified_components: Dict[str, Tuple[bool, str]], error_code: str,
            anomalous_comp: str, prev_classified_components: Dict[str, Tuple[bool, str]]
    ) -> None:
        """
        Works through the unisolated components, i.e., performs fault isolation.

        :param unisolated_comps: unisolated components to work though
        :param explicitly_considered_links: list of explicitly considered links
        :param classified_components: dict of already checked components (in classification state)
        :param error_code: error code the original component suggestion was based on
        :param anomalous_comp: initial anomalous component (entry point)
        :param prev_classified_components: previously checked components (used to avoid redundant classifications)
        """
        while len(unisolated_comps) > 0:
            comp_to_be_checked = unisolated_comps.pop(0)
            if self.verbose:
                print(colored("\ncomponent to be checked: " + comp_to_be_checked, "green", "on_grey", ["bold"]))
            if comp_to_be_checked not in list(explicitly_considered_links.keys()):
                explicitly_considered_links[comp_to_be_checked] = []

            # did we find an anomaly for this component before?
            if comp_to_be_checked in classified_components or comp_to_be_checked in prev_classified_components:
                prev_found_anomaly = (comp_to_be_checked in classified_components
                                      and classified_components[comp_to_be_checked][0]
                                      or comp_to_be_checked in prev_classified_components
                                      and prev_classified_components[comp_to_be_checked][0])
                if self.verbose:
                    print("already checked this component - anomaly:", prev_found_anomaly)
                if prev_found_anomaly:
                    self.handle_anomaly(
                        comp_to_be_checked, unisolated_comps, explicitly_considered_links, classified_components,
                        prev_classified_components
                    )
                continue
            # TODO: for now, we expect that all components can be diagnosed based on a sensor signal
            use_sensor_signal = True  # self.qt.query_sensor_signal_usage_by_suspect_component(comp_to_be_checked)[0]
            sim_model_data = {}
            if use_sensor_signal:
                if self.verbose:
                    print("use sensor signal..")
                classification_res = self.classify_component(
                    comp_to_be_checked, error_code, classified_components[anomalous_comp][1], sim_model_data
                )
                if classification_res is None:
                    anomaly = self.data_accessor.get_manual_judgement_for_component(comp_to_be_checked)
                    classification_id = self.instance_gen.extend_knowledge_graph_with_manual_inspection(
                        anomaly, classified_components[anomalous_comp][1], comp_to_be_checked
                    )
                else:
                    (anomaly, classification_id) = classification_res
                classified_components[comp_to_be_checked] = (anomaly, classification_id)
            else:
                anomaly = self.data_accessor.get_manual_judgement_for_component(comp_to_be_checked)
                classification_id = self.instance_gen.extend_knowledge_graph_with_manual_inspection(
                    anomaly, classified_components[anomalous_comp][1], comp_to_be_checked
                )
                classified_components[comp_to_be_checked] = (anomaly, classification_id)
            if anomaly:
                self.handle_anomaly(
                    comp_to_be_checked, unisolated_comps, explicitly_considered_links, classified_components,
                    prev_classified_components
                )
            self.log_classification_action(comp_to_be_checked, bool(anomaly), use_sensor_signal, classification_id)
            self.log_sim_classification_action(
                comp_to_be_checked, bool(anomaly), use_sensor_signal, classification_id, sim_model_data
            )

    @staticmethod
    def create_tmp_file_for_already_found_fault_paths(fault_paths: Dict[str, List[List[str]]]) -> None:
        """
        Creates a temporary file for already found fault paths.

        :param fault_paths: already found fault paths to be saved in session dir
        """
        if os.path.exists(SESSION_DIR + "/" + FAULT_PATH_TMP_FILE):
            with open(SESSION_DIR + "/" + FAULT_PATH_TMP_FILE, "r") as f:
                existing_data = json.load(f)
                for k in fault_paths:
                    if k in existing_data:  # update values
                        for v in fault_paths[k]:
                            if v not in existing_data[k]:
                                existing_data[k].append(v)
                    else:
                        existing_data[k] = fault_paths[k]
        else:
            existing_data = fault_paths
        with open(SESSION_DIR + "/" + FAULT_PATH_TMP_FILE, "w") as f:
            json.dump(existing_data, f, default=str)

    @staticmethod
    def save_already_found_anomaly_graph(anomaly_graph_key_str: str, res: List[List[str]]) -> None:
        if os.path.exists(SESSION_DIR + "/" + ANOMALY_GRAPH_TMP_FILE):
            with open(SESSION_DIR + "/" + ANOMALY_GRAPH_TMP_FILE, "r") as f:
                existing_data = json.load(f)
                existing_data[anomaly_graph_key_str] = res
        else:
            existing_data = {anomaly_graph_key_str: res}
        with open(SESSION_DIR + "/" + ANOMALY_GRAPH_TMP_FILE, "w") as f:
            json.dump(existing_data, f, default=str)

    @staticmethod
    def load_already_found_anomaly_graph_res() -> Dict[str, List[List[str]]]:
        path = SESSION_DIR + "/" + ANOMALY_GRAPH_TMP_FILE
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    @staticmethod
    def load_already_found_fault_paths() -> Dict[str, List[List[str]]]:
        """
        Loads the already found fault paths from the tmp file.

        :return: already found fault paths
        """
        path = SESSION_DIR + "/" + FAULT_PATH_TMP_FILE
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def find_unique_longest_paths_over_dict(self, paths: Dict[str, List[List[str]]]) -> Dict[str, List[List[str]]]:
        """
        Filters the identified fault paths, i.e., finds the unique longest paths over the specified dictionary.

        :param paths: dict of identified fault paths
        :return: filtered dict of fault paths
        """
        if self.verbose:
            print("FINDING UNIQUE PATHS:")
            for path in paths.keys():
                print(paths[path])

        # a fault path that is found based on several components is only stored
        # under the one it was first found with
        already_seen_paths = []
        for comp in paths.keys():
            updated_paths = []
            for path in paths[comp]:
                if path not in already_seen_paths:
                    updated_paths.append(path)
                    already_seen_paths.append(path)
            paths[comp] = updated_paths

        # another issue: subpaths coming from the file system (from previous iterations or states)
        all_paths = [path for comp in paths.keys() for path in paths[comp]]
        filtered_paths = self.find_unique_longest_paths(all_paths)
        final_dict = defaultdict(list)
        for comp in paths.keys():
            for path in paths[comp]:
                if path in filtered_paths:
                    final_dict[comp].append(path)

        if self.verbose:
            print("FINAL UNIQUE PATHS:")
        for comp in final_dict.keys():
            if self.verbose:
                print(final_dict[comp])
        return final_dict

    def find_paths_dfs(self, anomaly_graph, node, path=[]):
        if node in path:  # deal with cyclic relations
            return [path]
        path = path + [node]  # not using append() because it wouldn't create a new list
        if node not in anomaly_graph:
            return [path]
        paths = []
        for node in anomaly_graph[node]:
            paths.extend(self.find_paths_dfs(anomaly_graph, node, path))
        return paths

    def find_all_longest_paths(self, anomaly_graph):
        anomaly_graph_dict = self.load_already_found_anomaly_graph_res()
        anomaly_graph_key_str = ",".join(list(anomaly_graph.keys()))
        # avoid redundant computations
        if anomaly_graph_key_str in anomaly_graph_dict.keys():
            return anomaly_graph_dict[anomaly_graph_key_str]
        all_paths = []
        for path_src in anomaly_graph:
            all_paths.extend(self.find_paths_dfs(anomaly_graph, path_src))
        unique_paths = self.find_unique_longest_paths(all_paths)
        # save computed paths for anomaly graph, can be reused later
        self.save_already_found_anomaly_graph(anomaly_graph_key_str, unique_paths)
        return unique_paths

    @staticmethod
    def find_unique_longest_paths(paths):
        unique_paths = []
        paths_sorted = sorted(paths, key=len, reverse=True)
        for path in paths_sorted:
            if not any("-" + "-".join(list(path)) + "-" in "-" + "-".join(up) + "-" for up in unique_paths):
                unique_paths.append(list(path))
        return unique_paths

    def execute(self, userdata: smach.user_data.Remapper) -> str:
        """
        Execution of 'ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS' state.
        Implements the search in the causal graph (cause-effect network).

        :param userdata: input of state
        :return: outcome of the state ("isolated_problem" | "isolated_problem_remaining_error_codes")
        """
        self.log_state_info()
        # those are only the ones from the prev classification state, not all
        classified_components = self.retrieve_already_checked_components(userdata.classified_components)
        # only record those that haven't been recorded before - read already classified components
        with open(SESSION_DIR + "/" + CLASSIFICATION_LOG_FILE, "r") as f:
            log_file = json.load(f)
            prev_classified_components = {}
            # component name mapped to classification dict
            for classification in log_file:
                comp = list(classification.keys())[0]
                pred = classification[comp]
                classification_id = classification["Classification ID"]
                prev_classified_components[comp] = (pred, classification_id)

        # the components coming from the classification state are not supposed to be handled identical to the ones
        # classified before, the ones classified before are just to be considered as a lookup for classifications,
        # not triggering new classifications
        if self.verbose:
            print("ALREADY CHECKED COMPONENTS - CLASSIFICATION STATE:")
            print(len(list(classified_components.keys())))
            print("ALREADY CHECKED COMPONENTS - PREV ITERATIONS:")
            print(prev_classified_components.keys())

        anomalous_paths = {}
        if self.verbose:
            print(colored("constructing causal graph, i.e., subgraph of structural component knowledge..\n",
                          "green", "on_grey", ["bold"]))
        # complete_graphs = {comp: self.construct_complete_graph({}, [comp])
        #                   for comp in classified_components.keys() if classified_components[comp][0]}
        explicitly_considered_links = {}
        # self.visualize_initial_graph(anomalous_paths, complete_graphs, explicitly_considered_links)

        # load potential previous paths from session files
        already_found_fault_paths = self.load_already_found_fault_paths()

        # important to compare to userdata here to not have a dictionary of changed size during iteration
        for class_id in userdata.classified_components:
            anomalous_comp = self.retrieve_sus_comp(class_id)
            # anomalous component is a new component (just classified in prev state)
            if not classified_components[anomalous_comp][0]:
                continue  # already classified and no anomaly
            if self.verbose:
                print(colored("isolating " + anomalous_comp + "..", "green", "on_grey", ["bold"]))
            affecting_components = self.qt.query_affected_by_relations_by_suspect_component(anomalous_comp)

            if anomalous_comp not in list(explicitly_considered_links.keys()):
                explicitly_considered_links[anomalous_comp] = affecting_components.copy()
            else:
                explicitly_considered_links[anomalous_comp] += affecting_components.copy()

            if self.verbose:
                print("component potentially affected by:", affecting_components)
            unisolated_components = affecting_components
            self.work_through_unisolated_components(
                unisolated_components, explicitly_considered_links, classified_components,
                self.read_error_code_suggestion(anomalous_comp), anomalous_comp, prev_classified_components
            )
            if self.verbose:
                print("explicitly considered links:", explicitly_considered_links)
            edges = []
            for k in explicitly_considered_links.keys():
                if (k in classified_components and classified_components[k][0]
                        or k in prev_classified_components and prev_classified_components[k][0]):  # k has anomaly
                    for v in explicitly_considered_links[k]:
                        if (v in classified_components and classified_components[v][0]
                                or v in prev_classified_components and prev_classified_components[v][
                                    0]):  # v has anomaly
                            edges.append(k + " -> " + v)

            edges = edges[::-1]  # has to be reversed, affected-by direction
            # create adjacency lists
            anomaly_graph = defaultdict(list)
            for edge in edges:
                start, end = edge.split(' -> ')
                anomaly_graph[start].append(end)

            fault_paths = self.find_all_longest_paths(anomaly_graph)

            # handle one-component-paths
            all_previous_paths = list(already_found_fault_paths.values())
            if len(all_previous_paths) > 0:
                all_previous_paths = all_previous_paths[0][0]
            for k in explicitly_considered_links.keys():
                if ((k in classified_components and classified_components[k][0]
                     or k in prev_classified_components and prev_classified_components[k][0])
                        and k not in " ".join(edges)):  # unconsidered anomaly
                    if not any(k in i for i in all_previous_paths):  # also not part of previous paths
                        fault_paths.append([k])

            anomalous_paths[anomalous_comp] = fault_paths
        # visualizations = self.gen_causal_graph_visualizations(
        #     anomalous_paths, complete_graphs, explicitly_considered_links
        # )
        # self.data_provider.provide_causal_graph_visualizations(visualizations)
        remaining_error_code_instances = util.load_error_code_instances()
        if self.verbose:
            print("REMAINING error codes:", remaining_error_code_instances)
        if len(remaining_error_code_instances) > 0:
            self.create_tmp_file_for_already_found_fault_paths(anomalous_paths)  # write anomalous paths to session file
            self.data_provider.provide_state_transition(StateTransition(
                "ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS", "SELECT_UNUSED_ERROR_CODE",
                "isolated_problem_remaining_error_codes"
            ))
            return "isolated_problem_remaining_error_codes"

        # merge dictionaries (already found + new ones)
        for k in anomalous_paths:
            if k in already_found_fault_paths:  # update values
                for v in anomalous_paths[k]:
                    if v not in already_found_fault_paths[k]:
                        already_found_fault_paths[k].append(v)
            else:
                already_found_fault_paths[k] = anomalous_paths[k]

        userdata.fault_paths = self.find_unique_longest_paths_over_dict(already_found_fault_paths)
        self.data_provider.provide_state_transition(StateTransition(
            "ISOLATE_PROBLEM_CHECK_EFFECTIVE_RADIUS", "PROVIDE_DIAG_AND_SHOW_TRACE", "isolated_problem"
        ))
        return "isolated_problem"
