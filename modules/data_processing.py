from numba import njit, float64
from numba.experimental import jitclass

spec_input = [
    ('climatic_forcing',  float64[:, :]),
]

@jitclass(spec_input)
class inputs_data:
    def __init__(self, climatic_forcing):
        self.climatic_forcing = climatic_forcing

class data_processing:
    def generate_inputs(self,
                        ):
        """
        生成模型输入数据
        """
        self.model_inputs = inputs_data(self.basin_data[['P', 'E']][self.DATA_START : self.DATA_END].values)
        return self.model_inputs
    
    def generate_observations(self,
                              ):
        """
        生成模型观测数据
        """
        self.observations = self.basin_data['Qmm'][self.DATA_START : self.DATA_END].values
        return self.observations