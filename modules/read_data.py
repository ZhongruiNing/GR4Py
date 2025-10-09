import os
import json
import pandas as pd

from typing import Tuple

class read_data:
    def _read_global_conf(self) -> Tuple:
        
        # ------------------------------------------------------------------------------
        # 读取配置文件的函数
        # ------------------------------------------------------------------------------
        
        filepath = "conf/global_conf.json"
        # 检查文件是否存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"配置文件 {filepath} 不存在，请检查！")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查必要字段是否存在
        required_fields = ['basin_name', 'data_res', 'data_start', 'data_end']
        for field in required_fields:
            if field not in data:
                raise KeyError(f"配置文件缺少字段: {field}")
        
        # 解析json文件读取字段
        self.BASIN_NAME = data['basin_name']
        self.DATA_RES   = data['data_res']
        self.DATA_START = data['data_start']
        self.DATA_END   = data['data_end']
        
        return self.BASIN_NAME, self.DATA_RES, self.DATA_START, self.DATA_END

    def _read_data(self) -> pd.DataFrame:
        # ------------------------------------------------------------------------------
        # 读取流域数据的函数
        # ------------------------------------------------------------------------------
        
        # 数据文件
        data_filepath = f"data/{self.BASIN_NAME}.txt"
        if not os.path.exists(data_filepath):
            raise FileNotFoundError(f"数据文件 {data_filepath} 不存在，请检查！")
        
        self.basin_data = pd.read_csv(data_filepath, sep="\t", parse_dates=['Time'], index_col='Time')
        self.basin_data = self.basin_data[self.DATA_START : self.DATA_END]
        self.dates      = self.basin_data.index
        
        return self.basin_data
    
    def _read_calibration_conf(self) -> Tuple:
        
        # ------------------------------------------------------------------------------
        # 读取校准参数文件的函数
        # ------------------------------------------------------------------------------
        filepath = f"conf/calibration_conf.json"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"校准参数文件 {filepath} 不存在，请检查！")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        required_fields = ['max_iter', 'n_complex']
        for field in required_fields:
            if field not in params:
                raise KeyError(f"校准参数文件缺少字段: {field}")
        
        self.max_iter  = params['max_iter']
        self.n_complex = params['n_complex']
        
        return self.max_iter, self.n_complex
    
    def _read_params_bounds(self) -> Tuple:
        # ------------------------------------------------------------------------------
        # 读取参数边界文件的函数
        # ------------------------------------------------------------------------------
        filepath = f"conf/params_bounds.json"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"参数边界文件 {filepath} 不存在，请检查！")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            params_bounds = json.load(f)

        if self.FUN_MOD not in params_bounds:
            raise KeyError(f"参数边界文件中不包含模型 {self.FUN_MOD} 的参数边界，请检查！")
        
        params_bounds = params_bounds[self.FUN_MOD]
        
        required_fields = ['param_names', 'lower_bounds', 'upper_bounds']
        for field in required_fields:
            if field not in params_bounds:
                raise KeyError(f"参数边界文件缺少字段: {field}")
        
        self.param_names   = params_bounds['param_names']
        self.lower_bounds  = params_bounds['lower_bounds']
        self.upper_bounds  = params_bounds['upper_bounds']
        
        if not (len(self.param_names) == len(self.lower_bounds) == len(self.upper_bounds)):
            raise ValueError("参数名称、下边界和上边界的长度不一致，请检查！")
        if not len(self.param_names) == self.nparams:
            raise ValueError(f"参数数量 {len(self.param_names)} 与模型 {self.FUN_MOD} 的参数数量不匹配，请检查！")
        
        return self.param_names, self.lower_bounds, self.upper_bounds
