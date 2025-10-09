import numpy as np
import pandas as pd


class model_check:
    def __init__(self,
                 dates: pd.DatetimeIndex):
        self.dates = dates

    def _check_dates(self,
                     dates=None):
        if dates is None:
            dates = self.dates
        if dates is None:
            raise ValueError("dates不能为空，请指定！")
        # 检查日期输入格式
        # 若日期输入为列表，则转换为 DatetimeIndex，列表数字代表自1970-01-01的ns秒数
        if isinstance(dates, list):
            dates = pd.DatetimeIndex(dates)
        # 若日期输入不为列表且不为 DatetimeIndex，则尝试转换为 DatetimeIndex
        elif not isinstance(dates, pd.DatetimeIndex):
            try:
                dates = pd.to_datetime(dates)
                if not isinstance(dates, pd.DatetimeIndex):
                    dates = pd.DatetimeIndex([dates])
            except:
                raise ValueError("dates格式错误，必须可以转换为pd.DatetimeIndex，请检查！")
        # 判断日期序列是否存在重复值
        if dates.duplicated().any():
            raise ValueError("dates中存在重复值，请检查！")
        # 返回处理后的日期序列
        self.dates = dates
        return dates
    
    def _feat_models(self):
        # ------------------------------------------------------------------------------
        # 返回支持的模型名称列表
        # ------------------------------------------------------------------------------
        self.feat_models = pd.read_csv("conf/feat_model_GR.txt", sep="\t", header=0)
        return self.feat_models
    
    def _check_model(self):
        # ------------------------------------------------------------------------------
        # 检查输入模型是否合法
        # ------------------------------------------------------------------------------
        # 读取所有支持的模型
        self._feat_models()
        # 判断输入模型是否合法
        name_fun_mode = self.feat_models['NameMod'].values
        id_mod = np.where(name_fun_mode == self.FUN_MOD)[0]
        if len(id_mod) == 0:
            raise ValueError(f"模型 {self.FUN_MOD} 不可用. 可用模型为: {name_fun_mode.tolist()}")
        res = self.feat_models.iloc[id_mod[0]].to_dict()

        # 字段到数值的映射
        time_step_mapping = {
            'hourly': 1,
            'daily': 24,
            'monthly': np.array([28, 29, 30, 31]) * 24,
            'yearly': np.array([365, 366]) * 24
        }
        dates = self._check_dates(self.dates)
        time_diff = (dates[1:] - dates[:-1]).total_seconds() / 3600
        unique_diffs = np.unique(time_diff)
        if not np.all(np.isin(unique_diffs, time_step_mapping[res['TimeUnit']])):
            raise ValueError(f"输入数据中的时间步长不匹配预期的时间单位 '{res['TimeUnit']}'")
        
        NH_mapping = {
            'hourly': 480,
            'daily': 20,
            'monthly': 0,
            'yearly': 0
        }
        
        # 模型参数数值
        self.nparams = res['NbParam']
        self.NMISC   = int(res['NMISC'])
        self.NH      = NH_mapping[res['TimeUnit']]