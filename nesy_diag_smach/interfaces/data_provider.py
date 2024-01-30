#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from abc import ABC, abstractmethod
from typing import List

from PIL.Image import Image

from nesy_diag_smach.data_types.state_transition import StateTransition


class DataProvider(ABC):
    """
    Interface that defines the state machine's provision of intermediate results and diagnosis-relevant case data.
    """

    @abstractmethod
    def provide_causal_graph_visualizations(self, visualizations: List[Image]) -> None:
        """
        Provides causal graph visualizations.

        :param visualizations: causal graph visualizations
        """
        pass

    @abstractmethod
    def provide_heatmaps(self, heatmaps: Image, title: str) -> None:
        """
        Provides heatmap visualizations.

        :param heatmaps: heatmap visualizations
        :param title: title of the heatmap plot (component + result of classification + score)
        """
        pass

    @abstractmethod
    def provide_diagnosis(self, fault_paths: List[str]) -> None:
        """
        Provides the final diagnosis in the form of a set of fault paths.

        :param fault_paths: final diagnosis
        """
        pass

    @abstractmethod
    def provide_state_transition(self, state_transition: StateTransition) -> None:
        """
        Provides a transition performed by the state machine as part of a diagnostic process.

        :param state_transition: state transition (prev state -- (transition link) --> current state)
        """
        pass
