#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

from typing import List, Dict


class FaultContext:
    """
    Represents fault context data, which is communicated to the state machine.
    """

    def __init__(self, error_code_list: List[str], diag_subject_id: str) -> None:
        """
        Inits the fault context data.

        :param error_code_list: list of error codes
        :param diag_subject_id: identification number for the diagnostic subject
        """
        self.error_code_list = error_code_list
        self.diag_subject_id = diag_subject_id

    def get_json_representation(self) -> Dict[str, str]:
        """
        Returns a JSON representation of the fault context data.

        :return: JSON representation of fault context data
        """
        return {
            "error_code_list": self.error_code_list,
            "diag_subject_id": self.diag_subject_id
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the fault context data.

        :return: string representation of fault context data
        """
        return f"error code list: {self.error_code_list},\ndiag subject ID: {self.diag_subject_id}"
