#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from typing import List, Dict


class FaultContext:
    """
    Represents fault context data, which is communicated to the state machine.
    """

    def __init__(self, error_code_list: List[str], diag_entity_id: str) -> None:
        """
        Inits the fault context data.

        :param error_code_list: list of error codes
        :param diag_entity_id: identification number for the diagnostic entity
        """
        self.error_code_list = error_code_list
        self.diag_entity_id = diag_entity_id

    def get_json_representation(self) -> Dict[str, str]:
        """
        Returns a JSON representation of the fault context data.

        :return: JSON representation of fault context data
        """
        return {
            "error_code_list": self.error_code_list,
            "diag_entity_id": self.diag_entity_id
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the fault context data.

        :return: string representation of fault context data
        """
        return f"error code list: {self.error_code_list},\ndiag entity ID: {self.diag_entity_id}"
