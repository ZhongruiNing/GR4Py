import numpy as np
import sys
sys.path.append("../src")

from sceua import sceua
from utils import nash_sutcliffe_efficiency

# @njit
def cost_function(x, param, observation, fn_hm):
    simulation = fn_hm(x, param)
    KGE = nash_sutcliffe_efficiency(observation, simulation)
    CF = 1 - KGE
    return CF

class model_calibration:
    def calibrate(self):
        bounds     = np.array([self.lower_bounds, self.upper_bounds])
        max_iter   = self.max_iter
        n_complex  = self.n_complex
        params_num = self.nparams
        x_obs      = self.generate_inputs()
        y_obs      = self.generate_observations()
        bestx, _ = sceua(bounds, max_iter, n_complex, params_num, x_obs, y_obs, self.run_model, cost_function)
        self.bestx = bestx
        return bestx