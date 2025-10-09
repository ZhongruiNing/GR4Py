#------------------------------------------------------------------------------
#    Subroutines relative to the annual GR4J model
#------------------------------------------------------------------------------
# TITLE   : GR4Py
# PROJECT : GR4Py
# FILE    : pyrun_GR4J.py
#------------------------------------------------------------------------------
# AUTHORS
# Original code: Zhongrui Ning
#------------------------------------------------------------------------------
# Creation date: 2025
# Last modified: 09/20/2025
#------------------------------------------------------------------------------
# REFERENCES
# Perrin, C., Michel, C. and Andréassian, V. (2003). Improvement of a
# parsimonious model for streamflow simulation. Journal of Hydrology,
# 279(1-4), 275-289, doi: 10.1016/S0022-1694(03)00225-7.
#------------------------------------------------------------------------------
# Quick description of public procedures:
#         1. pyrun_GR4J
#         2. MOD_GR4J
#------------------------------------------------------------------------------

import numpy as np

from typing import Tuple
from numba import njit
from utils import UH1_D, UH2_D

@njit
def _MOD_GR4J(
        St:     np.ndarray,
        StUH1:  np.ndarray,
        StUH2:  np.ndarray,
        OrdUH1: np.ndarray,
        OrdUH2: np.ndarray,
        Param:  np.ndarray,
        P1:     float,
        E:      float,
        NH:     float,
        NMISC:  float
) -> Tuple[float, np.ndarray]:
    """
    使用GR4J模型在单个时间步（天）上计算流量
    
    Parameters:
    -----------
    st : np.ndarray
        时间步开始时库容的模型状态 [mm]
    StUH1 : np.ndarray
        时间步开始时单位线UH1的模型状态 [mm]
    StUH2 : np.ndarray
        时间步开始时单位线UH2的模型状态 [mm]
    OrdUH1 : np.ndarray
        UH1中的序列 [-]
    OrdUH2 : np.ndarray
        UH2中的序列 [-]
    Param : class
        模型参数 [各种单位]
    P1 : float
        时间步内的降雨量 [mm]
    E : float
        时间步内的潜在蒸散发量 [mm]
    
    Returns:
    --------
    Tuple[float, np.ndarray]
        Q : 时间步内流域出口的模拟流量 [mm/day]
        MISC : 时间步的模型输出 [mm or mm/day]
    """
    # 常数
    B = 0.9
    STORED_VAL = 25.62890625  # (9/4)**4
    
    # 初始化输出
    MISC = np.full(NMISC, -999.999)
    
    # 获取参数
    X1 = Param[0]  # 生产库容量 [mm]
    X2 = Param[1]  # 流域间交换系数 [mm/day]
    X3 = Param[2]  # 汇流库容量 [mm]
    X4 = Param[3]  # 单位水文线时间常数 [day]
    
    # 拦截和生产库模块
    if P1 <= E:
        EN = E - P1
        PN = 0.0
        WS = EN / X1
        if WS > 13.0:
            WS = 13.0

        # 速度优化的双曲正切计算
        expWS = np.exp(2.0 * WS)
        TWS = (expWS - 1.0) / (expWS + 1.0)
        Sr = St[0] / X1
        ER = St[0] * (2.0 - Sr) * TWS / (1.0 + (1.0 - Sr) * TWS)
        
        AE = ER + P1
        St[0] = St[0] - ER
        PS = 0.0
        PR = 0.0
    else:
        EN = 0.0
        AE = E
        PN = P1 - E
        WS = PN / X1
        if WS > 13.0:
            WS = 13.0

        # 速度优化的双曲正切计算
        expWS = np.exp(2.0 * WS)
        TWS = (expWS - 1.0) / (expWS + 1.0)
        Sr = St[0] / X1
        PS = X1 * (1.0 - Sr * Sr) * TWS / (1.0 + Sr * TWS)

        PR = PN - PS
        St[0] = St[0] + PS

    # 生产库渗透
    if St[0] < 0.0:
        St[0] = 0.0
    
    # 速度优化的渗透计算
    Sr = St[0] / X1
    Sr = Sr * Sr
    Sr = Sr * Sr
    PERC = St[0] * (1.0 - 1.0 / np.sqrt(np.sqrt(1.0 + Sr / STORED_VAL)))
    St[0] = St[0] - PERC

    PR = PR + PERC

    # 有效降雨分配到两个汇流分支
    PRUH1 = PR * B
    PRUH2 = PR * (1.0 - B)

    # UH1 卷积
    uh1_max = max(1, min(NH - 1, int(X4 + 1)))
    for k in range(uh1_max):
        StUH1[k] = StUH1[k + 1] + OrdUH1[k] * PRUH1
    StUH1[NH - 1] = OrdUH1[NH - 1] * PRUH1
    
    # UH2 卷积
    uh2_max = max(1, min(2 * NH - 1, 2 * int(X4 + 1)))
    for k in range(uh2_max):
        StUH2[k] = StUH2[k + 1] + OrdUH2[k] * PRUH2
    StUH2[2 * NH - 1] = OrdUH2[2 * NH - 1] * PRUH2
    
    # 潜在流域间半交换
    Rr = St[1] / X3
    EXCH = X2 * Rr * Rr * Rr * np.sqrt(Rr)
    
    # 汇流库
    AEXCH1 = EXCH
    if (St[1] + StUH1[0] + EXCH) < 0.0:
        AEXCH1 = -St[1] - StUH1[0]
    St[1] = St[1] + StUH1[0] + EXCH
    if St[1] < 0.0:
        St[1] = 0.0
    
    # 速度优化的汇流库出流计算
    Rr = St[1] / X3
    Rr = Rr * Rr
    Rr = Rr * Rr
    QR = St[1] * (1.0 - 1.0 / np.sqrt(np.sqrt(1.0 + Rr)))
    St[1] = St[1] - QR
    
    # 直接分支径流 QD
    AEXCH2 = EXCH
    if (StUH2[0] + EXCH) < 0.0:
        AEXCH2 = -StUH2[0]
    QD = max(0.0, StUH2[0] + EXCH)
    
    # 总径流
    Q = QR + QD
    if Q < 0.0:
        Q = 0.0
    
    # 存储变量到 MISC 数组
    MISC[0] = E           # PE     观测潜在蒸发量 [mm/day]
    MISC[1] = P1          # Precip 观测总降水量 [mm/day]
    MISC[2] = St[0]       # Prod   生产库水位 [mm]
    MISC[3] = PN          # Pn     净降雨 [mm/day]
    MISC[4] = PS          # Ps     填充生产库的部分 [mm/day]
    MISC[5] = AE          # AE     实际蒸发量 [mm/day]
    MISC[6] = PERC        # PERC   渗透量 [mm/day]
    MISC[7] = PR          # PR     PR=PN-PS+PERC [mm/day]
    MISC[8] = StUH1[0]   # Q9     UH1 出流 [mm/day]
    MISC[9] = StUH2[0]   # Q1     UH2 出流 [mm/day]
    MISC[10] = St[1]      # Rout   汇流库水位 [mm]
    MISC[11] = EXCH       # EXCH   潜在半交换 [mm/day]
    MISC[12] = AEXCH1     # AEXCH1 分支1实际交换 [mm/day]
    MISC[13] = AEXCH2     # AEXCH2 分支2实际交换 [mm/day]
    MISC[14] = AEXCH1 + AEXCH2  # AEXCH 总实际交换 [mm/day]
    MISC[15] = QR         # QR     汇流库出流 [mm/day]
    MISC[16] = QD         # QD     UH2分支交换后出流 [mm/day]
    MISC[17] = Q          # Qsim   模拟出口流量 [mm/day]
    
    return Q, MISC

@njit
def _pyrun_GR4J(
        inputs_data,
        # inputs_precip:  np.ndarray,
        # inputs_pe:      np.ndarray, 
        Param:          np.ndarray,
        NH:             int,
        NMISC:          int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GR4J模型的主运行函数，初始化GR4J，获取参数，在每个时间步调用MOD_GR4J子程序，并存储最终状态
    
    Parameters:
    -----------
    inputs_data : class
        所有输入数据
    inputs_data.inputs_precip : np.ndarray
        输入的总降水时间序列 [mm/day]
    inputs_data.inputs_pe : np.ndarray
        输入的潜在蒸散发时间序列 [mm/day]
    Param : np.ndarray
        模型参数数组 [X1, X2, X3, X4]
        X1: 生产库容量 [mm]
        X2: 流域间交换系数 [mm/day]
        X3: 汇流库容量 [mm]
        X4: 单位水文线时间常数 [day]
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Outputs : 输出序列
        state_end : 模型运行结束时的状态变量
    """

    inputs_precip = inputs_data.climatic_forcing[:, 0]
    inputs_pe     = inputs_data.climatic_forcing[:, 1]

    l_inputs  = len(inputs_precip)
    
    # 初始化
    St = np.zeros(3)  # 3个主要状态变量
    StUH1 = np.zeros(NH)  # UH1状态
    StUH2 = np.zeros(2 * NH)  # UH2状态
    
    # # 使用StateStart初始化模型状态
    # St[0] = state_start[0]  # 产流库水位
    # St[1] = state_start[1]  # 汇流库水位
    # St[2] = state_start[2]  # 指数库水位
    
    # # 初始化单位线状态
    # for i in range(NH):
    #     StUH1[i] = state_start[7 + i]
    # for i in range(2 * NH):
    #     StUH2[i] = state_start[7 + NH + i]
    
    # 计算单位线序列
    D = 2.5
    OrdUH1 = UH1_D(Param[3], D)
    OrdUH2 = UH2_D(Param[3], D)
    
    # 初始化输出数组
    Outputs = np.full((l_inputs, NMISC), -999.999)
    
    # 时间循环
    for k in range(l_inputs):
        P1 = inputs_precip[k]
        E  = inputs_pe[k]
        
        # 单时间步模型计算
        Q, MISC = _MOD_GR4J(St, StUH1, StUH2, OrdUH1, OrdUH2, Param, P1, E, NH, NMISC)

        Outputs[k, :] = MISC
    
    # 构建最终状态
    # state_end = np.full(7 + NH + 2 * NH, -999.999)
    # state_end[0] = St[0]
    # state_end[1] = St[1]
    # state_end[7 : 7 + NH] = StUH1
    # state_end[7 + NH : 7 + NH + 2 * NH] = StUH2
    
    return Outputs