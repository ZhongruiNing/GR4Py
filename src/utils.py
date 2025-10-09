import numpy as np
import math
from numba import njit

@njit
def SS1_D(i: int,
           C: float,
           D: float
           ) -> float:
    """
    计算 GR 单位线 UH1 的 S 曲线（累积单位线曲线）值的函数。

    参数:
        i (int): 时间步长（Fortran 中的 I）。
        C (float): 实数，时间常数。
        D (float): 实数，指数。

    返回:
        float: 时间步长 i 对应的 S 曲线值。
    """
    if i <= 0:
        return 0.0
    if i < C:
        return (i / C) ** D
    # This corresponds to the case where i >= c
    return 1.0

@njit
def SS2_D(i: int,
           C: float,
           D: float
           ) -> float:
    """
    计算 GR 单位线 UH2 的 S 曲线（累积单位线曲线）值的函数。

    参数:
        i (int): 时间步长（Fortran 中的 I）。
        C (float): 实数，时间常数。
        D (float): 实数，指数。

    返回:
        float: 时间步长 i 对应的 S 曲线值。
    """
    if i <= 0:
        return 0.0
    if i <= C:
        return 0.5 * (i / C) ** D
    if i < 2.0 * C:
        return 1.0 - 0.5 * (2.0 - i / C) ** D
    # This corresponds to the case where i >= 2*c
    return 1.0

@njit
def UH1_D(C: float,
           D: float
           ) -> np.ndarray:
    """
    使用 S 曲线 SS1 的连续差分计算每日 GR 单位线 UH1 的纵坐标。

    参数:
        C (float): 实数，时间常数。
        D (float): 实数，指数。

    返回:
        np.ndarray: 一个形状为 (NH,) 的 numpy 数组，包含离散单位线的 NH 个纵坐标。
    """
    NH = 20  # From Fortran parameter :: NH=480
    ord_uh1 = np.zeros(NH, dtype=np.float64)  # Initialize with zeros, Fortran default is not always zero.

    for i in range(NH):
        # Python uses 0-based indexing, so we store the result for time i at index i-1.
        ord_uh1[i] = SS1_D(i + 1, C, D) - SS1_D(i, C, D)
    return ord_uh1

@njit
def UH2_D(C: float,
           D: float
           ) -> np.ndarray:
           
    """
    使用 S 曲线 SS2 的连续差分计算每日 GR 单位线 UH2 的纵坐标。

    参数:
        C (float): 实数，时间常数。
        D (float): 实数，指数。

    返回:
        np.ndarray: 一个形状为 (2*NH,) 的 numpy 数组，包含离散单位线的 2*NH 个纵坐标。
    """
    NH = 20  # From Fortran parameter :: NH=480
    ord_uh2 = np.zeros(2 * NH, dtype=np.float64)  # Initialize with zeros

    # The length of UH2 is 2*NH
    for i in range(2 * NH):
        # Python uses 0-based indexing, so we store the result for time i at index i.
        ord_uh2[i] = SS2_D(i + 1, C, D) - SS2_D(i, C, D)
    return ord_uh2

@njit
def SS1_H(i: int, C: float, D: float) -> float:
    """
    计算 GR 单位线 UH1 的 S 曲线（累积单位线曲线）值的函数。

    参数:
        i (int): 时间步长（Fortran 中的 I）。
        C (float): 实数，时间常数。
        D (float): 实数，指数。

    返回:
        float: 给定时间步长的 S 曲线值（Fortran 中的 SS1_H）。
    """
    # In Fortran, FI = I was used. In Python, we can directly use i.

    if i <= 0:
        return 0.0
    elif i < C:
        return (i / C) ** D
    else:  # i >= C
        return 1.0
    
@njit
def SS2_H(i: int, C: float, D: float) -> float:
    """
    计算 GR 单位线 UH2 的 S 曲线（累积单位线曲线）值的函数。

    参数:
        i (int): 时间步长（Fortran 中的 I）。
        C (float): 实数，时间常数。
        D (float): 实数，指数。

    返回:
        float: 给定时间步长的 S 曲线值（Fortran 中的 SS2_H）。
    """
    # In Fortran, FI = I was used. In Python, we can directly use i.

    if i <= 0:
        return 0.0
    elif i <= C:
        return 0.5 * (i / C) ** D
    elif i < 2.0 * C:
        return 1.0 - 0.5 * (2.0 - i / C) ** D
    else:  # i >= 2.0 * C
        return 1.0

@njit  
def UH1_H(C: float, D: float) -> np.ndarray:
    """
    子程序通过对 S 曲线 SS1 进行连续差分计算小时 GR 单位线 UH1 的纵坐标。

    参数:
        C (float): 实数，时间常数。
        D (float): 实数，指数。

    返回:
        numpy.ndarray: 长度为 NH (480) 的离散单位线纵坐标向量 (Fortran 中的 OrdUH1)。
    """
    NH = 480  # From Fortran parameter :: NH=480
    ord_uh1 = np.zeros(NH, dtype=np.float64)  # Initialize with zeros, Fortran default is not always zero.

    for i in range(NH):  # Python's range(NH) goes from 0 to NH-1
        # Fortran loop was I=1,NH. Here we use 0-indexed array,
        # but the SS1_H function takes the 'Fortran' time-step (1-based index).
        # So, we pass (i+1) for the current step and i for the previous step.
        ord_uh1[i] = SS1_H(i + 1, C, D) - SS1_H(i, C, D)
    return ord_uh1

@njit
def UH2_H(C: float, D: float) -> np.ndarray:
    """
    子程序通过对 S 曲线 SS2 进行连续差分计算小时 GR 单位线 UH2 的纵坐标。

    参数:
        C (float): 实数，时间常数。
        D (float): 实数，指数。

    返回:
        numpy.ndarray: 长度为 2*NH (960) 的离散单位线纵坐标向量 (Fortran 中的 OrdUH2)。
    """
    NH = 480  # From Fortran parameter :: NH=480
    ord_uh2 = np.zeros(2 * NH, dtype=np.float64)  # Initialize with zeros

    for i in range(2 * NH):  # Python's range(2*NH) goes from 0 to 2*NH-1
        # Fortran loop was I=1,2*NH. Here we use 0-indexed array,
        # but the SS2_H function takes the 'Fortran' time-step (1-based index).
        # So, we pass (i+1) for the current step and i for the previous step.
        ord_uh2[i] = SS2_H(i + 1, C, D) - SS2_H(i, C, D)

    return ord_uh2

@njit
def _cal_mean(x):
    """计算均值"""
    total = 0.0
    for i in range(x.shape[0]):
        total += x[i]
    return total / x.shape[0]

@njit
def _get_min_along_axis0(x):
    """获取矩阵每列的最小值"""
    min_values = np.empty(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        min_values[j] = x[:, j].min()
    return min_values

@njit 
def _get_max_along_axis0(x):
    """获取矩阵每列的最大值"""
    max_values = np.empty(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        max_values[j] = x[:, j].max()
    return max_values

@njit
def _get_valid_data(observation, simulation):
    n = len(observation)
    # 统计有效数据
    valid_obs = np.empty(n, dtype=np.float64)
    valid_sim = np.empty(n, dtype=np.float64)
    count = 0
    for i in range(n):
        if not math.isnan(observation[i]) and not math.isnan(simulation[i]):
            valid_obs[count] = observation[i]
            valid_sim[count] = simulation[i]
            count += 1
    return valid_obs[:count], valid_sim[:count], count

@njit
def nash_sutcliffe_efficiency(observation, simulation):
    """计算 Nash-Sutcliffe 效率系数 (NSE)"""
    valid_obs, valid_sim, count = _get_valid_data(observation, simulation)
    if count == 0:
        return np.nan
    # 计算均值
    obs_mean = _cal_mean(valid_obs[:count])
    # 计算 NSE 公式
    num = 0.0
    denom = 0.0
    for i in range(count):
        num += (valid_obs[i] - valid_sim[i]) ** 2
        denom += (valid_obs[i] - obs_mean) ** 2

    return 1 - num / denom if denom != 0 else np.nan  # 避免除零错误

@njit
def relative_error(observation, simulation):
    """计算相对误差"""
    valid_obs, valid_sim, count = _get_valid_data(observation, simulation)
    if count == 0:
        return np.nan
    obs_mean = _cal_mean(valid_obs)
    sim_mean = _cal_mean(valid_sim)
    RE = (sim_mean - obs_mean) / obs_mean
    return RE

@njit
def pearson_correlation_coefficient(observation, simulation):
    """计算 Pearson 相关系数"""
    valid_obs, valid_sim, count = _get_valid_data(observation, simulation)
    if count == 0:
        return np.nan
    # 计算均值
    obs_mean = _cal_mean(valid_obs)
    sim_mean = _cal_mean(valid_sim)
    # 计算标准差
    obs_std = 0.0
    sim_std = 0.0
    for i in range(count):
        obs_std += (valid_obs[i] - obs_mean) ** 2
        sim_std += (valid_sim[i] - sim_mean) ** 2
    obs_std = np.sqrt(obs_std / count)
    sim_std = np.sqrt(sim_std / count)
    # 计算相关系数
    cov = 0.0
    for i in range(count):
        cov += (valid_obs[i] - obs_mean) * (valid_sim[i] - sim_mean)
    r = cov / (count * obs_std * sim_std) if (obs_std * sim_std) != 0 else 0.0
    return r

@njit
def kling_gupta_efficiency(observation, simulation):
    """计算 Kling-Gupta 效率系数 (KGE)"""
    valid_obs, valid_sim, count = _get_valid_data(observation, simulation)
    if count == 0:
        return np.nan
    # 计算均值
    obs_mean = _cal_mean(valid_obs)
    sim_mean = _cal_mean(valid_sim)
    # 计算标准差
    obs_std = 0.0
    sim_std = 0.0
    for i in range(count):
        obs_std += (valid_obs[i] - obs_mean) ** 2
        sim_std += (valid_sim[i] - sim_mean) ** 2
    obs_std = np.sqrt(obs_std / count)
    sim_std = np.sqrt(sim_std / count)
    # 计算相关系数
    cov = 0.0
    for i in range(count):
        cov += (valid_obs[i] - obs_mean) * (valid_sim[i] - sim_mean)
    r = cov / (count * obs_std * sim_std) if (obs_std * sim_std) != 0 else 0.0
    # 计算 alpha 和 beta
    alpha = sim_std / obs_std if obs_std != 0 else np.nan
    beta = sim_mean / obs_mean if obs_mean != 0 else np.nan
    # 计算 KGE
    kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    return kge