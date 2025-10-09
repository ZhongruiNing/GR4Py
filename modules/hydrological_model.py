import numpy as np
import sys
sys.path.append("../src")

from typing import Tuple
from read_data import read_data
from data_processing import data_processing
from model_check import model_check
from model_calibration import model_calibration
from pyrun_GR4J import _pyrun_GR4J
from pyrun_GR6J import _pyrun_GR6J
from pyrun_GR4H import _pyrun_GR4H

class pyrun_model:
    def run_GR4J(self,
                 inputs_data,
                 params: np.ndarray,
                 ) -> Tuple[np.ndarray, np.ndarray]:
        outputs = _pyrun_GR4J(
            inputs_data,
            params,
            self.NH,
            self.NMISC,
        )
        return outputs[:, -1]
    
    def run_GR4H(self,
                 inputs_data,
                 params: np.ndarray,
                 ) -> Tuple[np.ndarray, np.ndarray]:
        outputs = _pyrun_GR4H(
            inputs_data,
            params,
            self.NH,
            self.NMISC,
        )
        return outputs[:, -1]
    
    def run_GR6J(self,
                 inputs_data,
                 params: np.ndarray,
                 ) -> Tuple[np.ndarray, np.ndarray]:
        outputs = _pyrun_GR6J(
            inputs_data,
            params,
            self.NH,
            self.NMISC,
        )
        return outputs[:, -1]
    
class hydrological_model(read_data, data_processing, model_check, model_calibration, pyrun_model):
    """
    水文模型基类
    """
    def __init__(self,
                 FUN_MOD: str):
        self.FUN_MOD = FUN_MOD

    def run_model(self, inputs_data, params):
        if self.FUN_MOD == "GR4J":
            outputs = self.run_GR4J(inputs_data, params)
        elif self.FUN_MOD == "GR6J":
            outputs = self.run_GR6J(inputs_data, params)
        elif self.FUN_MOD == "GR4H":
            outputs = self.run_GR4H(inputs_data, params)
        else:
            raise ValueError(f"Unsupported model function: {self.FUN_MOD}")
        return outputs