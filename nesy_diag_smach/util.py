#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import json
from typing import Dict, List, Tuple

import numpy as np
from oscillogram_classification import cam
from oscillogram_classification import preprocess
from tensorflow import keras
from termcolor import colored

from nesy_diag_smach.config import SESSION_DIR, ERROR_CODE_TMP_FILE
from nesy_diag_smach.data_types.sensor_data import SensorData


def validate_keras_model(model: keras.models.Model) -> None:
    """
    The provided model is expected to be a `Keras` model satisfying the following assumptions:
        - input_shape: (None, len_of_ts, 1)
        - output_shape: (None, 1)
    Thus, in both cases we have a variable batch size due to `None`. For the input we expect a list of scalars and
    for the output exactly one scalar.

    :param model: model to be validated

    :raise ValueError: if the model shape doesn't match the expected one
    """
    in_shape = model.input_shape
    out_shape = model.output_shape
    expected_in_shape = (None, in_shape[1], 1)
    expected_out_shape_one_neuron = (None, 1)
    expected_out_shape_two_neurons = (None, 2)

    if len(in_shape) != len(expected_in_shape) or any(dim1 != dim2 for dim1, dim2 in zip(in_shape, expected_in_shape)):
        raise ValueError(f"unexpected input shape - expected: {expected_in_shape}, got: {in_shape}")

    if len(out_shape) != len(expected_out_shape_one_neuron) \
            or (any(dim1 != dim2 for dim1, dim2 in zip(out_shape, expected_out_shape_one_neuron))
                and any(dim1 != dim2 for dim1, dim2 in zip(out_shape, expected_out_shape_two_neurons))):
        raise ValueError(f"unexpected output shape: {out_shape}")


def preprocess_time_series_based_on_model_meta_info(
        model_meta_info: Dict[str, str], values: List[float], verbose: bool = True
) -> List[float]:
    """
    Preprocesses the time series based on model metadata (e.g., normalization method).
    The preprocessing always depends on the trained model that is going to be applied.
    Therefore, this kind of meta information has to be saved for each trained model.

    :param model_meta_info: metadata for the trained model (e.g., normalization method)
    :param values: raw input (time series values)
    :param verbose: whether terminal logs should be enabled
    :return: preprocessed input
    """
    if verbose:
        print("model meta info:", model_meta_info)
    if model_meta_info["normalization_method"] == "z_norm":
        return preprocess.z_normalize_time_series(values)
    elif model_meta_info["normalization_method"] == "min_max_norm":
        return preprocess.min_max_normalize_time_series(values)
    elif model_meta_info["normalization_method"] == "dec_norm":
        return preprocess.decimal_scaling_normalize_time_series(values, 2)
    elif model_meta_info["normalization_method"] == "log_norm":
        return preprocess.logarithmic_normalize_time_series(values, 10)
    return values


def no_trained_model_available(sensor_data: SensorData, suggestion_list: Dict[str, Tuple[str, bool]]) -> None:
    """
    Handles cases where no trained model is available for the specified component.

    :param sensor_data: sensor data
    :param suggestion_list: suspect components suggested for analysis {comp_name: (reason_for, sensor_usage)}
    """
    print("no trained model available for the signal (component) to be classified:", sensor_data.comp_name)
    print("adding it to the list of components to be verified manually..")
    suggestion_list[sensor_data.comp_name] = suggestion_list[sensor_data.comp_name][0], False


def invalid_model(sensor_data: SensorData, suggestion_list: Dict[str, Tuple[str, bool]], error: ValueError) -> None:
    """
    Handles cases where no valid trained model is available for the specified component.

    :param sensor_data: sensor data
    :param suggestion_list: suspect components suggested for analysis {comp_name: (reason_for, sensor_usage)}
    :param error: thrown value error
    """
    print("invalid model for the signal (component) to be classified:", sensor_data.comp_name)
    print("error:", error)
    print("adding it to the list of components to be verified manually..")
    suggestion_list[sensor_data.comp_name] = suggestion_list[sensor_data.comp_name][0], False


def construct_net_input(model: keras.models.Model, values: List[float]) -> np.ndarray:
    """
    Constructs / reshapes the input for the trained neural net model.

    :param model: trained neural net model
    :param values: input values (time series) to be reshaped
    :return: constructed / reshaped input
    """
    net_input_size = model.layers[0].output_shape[0][1]
    assert net_input_size == len(values)
    net_input = np.asarray(values).astype('float32')
    return net_input.reshape((net_input.shape[0], 1))


def log_anomaly(pred_value: float) -> None:
    """
    Logs anomalies.

    :param pred_value: prediction (output value of neural net)
    """
    print("#####################################")
    print(colored("--> ANOMALY DETECTED (" + str(pred_value) + ")", "green", "on_grey", ["bold"]))
    print("#####################################")


def log_regular(pred_value: float) -> None:
    """
    Logs regular cases, i.e., non-anomalies.

    :param pred_value: prediction (output value of neural net)
    """
    print("#####################################")
    print(colored("--> NO ANOMALIES DETECTED (" + str(pred_value) + ")", "green", "on_grey", ["bold"]))
    print("#####################################")


def gen_heatmaps(net_input: np.ndarray, model: keras.models.Model, prediction: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generates the heatmaps (visual explanations) for the classification.

    :param net_input: input sample
    :param model: trained classification model
    :param prediction: prediction, i.e., outcome of the model
    :return: dictionary of different heatmaps
    """
    return {"tf-keras-gradcam": cam.tf_keras_gradcam(np.array([net_input]), model, prediction),
            "tf-keras-gradcam++": cam.tf_keras_gradcam_plus_plus(np.array([net_input]), model, prediction),
            # TODO: not using that for the moment, leads to unwanted progress logs to terminal
            # "tf-keras-scorecam": cam.tf_keras_scorecam(np.array([net_input]), model, prediction),
            "tf-keras-layercam": cam.tf_keras_layercam(np.array([net_input]), model, prediction)}


def load_error_code_instances() -> List[str]:
    """
    Loads the error code instances from the tmp file.

    :return: list of error codes
    """
    with open(SESSION_DIR + "/" + ERROR_CODE_TMP_FILE) as f:
        return json.load(f)['list']


def log_info(msg) -> None:
    """
    Custom logging to override defaults.

    :param msg: msg to be logged
    """
    pass


def log_warn(msg) -> None:
    """
    Custom logging to override defaults.

    :param msg: msg to be logged
    """
    pass


def log_debug(msg) -> None:
    """
    Custom logging to override defaults.

    :param msg: msg to be logged
    """
    pass


def log_err(msg) -> None:
    """
    Custom logging to override defaults.

    :param msg: msg to be logged
    """
    print("[ ERROR ] : " + str(msg))
