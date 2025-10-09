#------------------------------------------------------------------------------
#    Subroutines relative to the annual GR6J model
#------------------------------------------------------------------------------
# TITLE   : GR4Py
# PROJECT : GR4Py
# FILE    : pyrun_GR6J.py
#------------------------------------------------------------------------------
# AUTHORS
# Original code: Zhongrui Ning
#------------------------------------------------------------------------------
# Creation date: 2025
# LaSt modified: 09/20/2025
#------------------------------------------------------------------------------
# REFERENCES
# Pushpalatha, R., Perrin, C., Le Moine, N., Mathevet, T. and Andréassian, V.
# (2011). A downward Structural sensitivity analysis of hydrological models to
# improve low-flow simulation. Journal of Hydrology, 411(1-2), 66-76,
# doi: 10.1016/j.jhydrol.2011.09.034.
#------------------------------------------------------------------------------
# Quick description of public procedures:
#         1. pyrun_gr6j
#         2. MOD_GR6J
#------------------------------------------------------------------------------

import numpy as np
import math

from typing import Tuple
from numba import njit
from utils import UH1_D, UH2_D

@njit
def _MOD_GR6J(
    St:      np.ndarray,
    StUH1:  np.ndarray,
    StUH2:  np.ndarray,
    OrdUH1: np.ndarray,
    OrdUH2: np.ndarray,
    Param:   np.ndarray,
    P1:      float,
    E:       float,
    NH:      int,
    NMISC:   int,
) -> Tuple[float, np.ndarray]:
    """
    使用GR6J模型在单个时间步（天）上计算流量
    
    Parameters:
    -----------
    St : np.ndarray
        时间步开始时库容的模型状态 [mm]
    StUH1 : np.ndarray
        时间步开始时单位线UH1的模型状态 [mm]
    StUH2 : np.ndarray
        时间步开始时单位线UH2的模型状态 [mm]
    OrdUH1 : np.ndarray
        UH1中的序列 [-]
    OrdUH2 : np.ndarray
        UH2中的序列 [-]
    Param : np.ndarray
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
    C = 0.4
    STORED_VAL = 25.62890625  # (9/4)**4
    
    # 参数
    A = Param[0]  # 产流库容量
    
    # 初始化
    MISC = np.full(NMISC, -999.999)
    
    # 产流库
    if P1 <= E:
        EN = E - P1
        PN = 0.0
        WS = EN / A
        if WS > 13:
            WS = 13.0
        
        # 加速计算
        expWS = math.exp(2.0 * WS)
        TWS = (expWS - 1.0) / (expWS + 1.0)
        Sr = St[0] / A
        ER = St[0] * (2.0 - Sr) * TWS / (1.0 + (1.0 - Sr) * TWS)
        
        AE = ER + P1
        St[0] = St[0] - ER
        PS = 0.0
        PR = 0.0
    else:
        EN = 0.0
        AE = E
        PN = P1 - E
        WS = PN / A
        if WS > 13:
            WS = 13.0
        
        # 加速计算
        expWS = math.exp(2.0 * WS)
        TWS = (expWS - 1.0) / (expWS + 1.0)
        Sr = St[0] / A
        PS = A * (1.0 - Sr * Sr) * TWS / (1.0 + Sr * TWS)

        PR = PN - PS
        St[0] = St[0] + PS

    # 产流库渗透
    if St[0] < 0:
        St[0] = 0.0
    
    # 加速计算
    Sr = St[0] / Param[0]
    Sr = Sr * Sr
    Sr = Sr * Sr
    PERC = St[0] * (1.0 - 1.0 / math.sqrt(math.sqrt(1.0 + Sr / STORED_VAL)))
    
    St[0] = St[0] - PERC
    PR = PR + PERC
    
    # 有效降雨分解为两个汇流分量
    PRUH1 = PR * B
    PRUH2 = PR * (1.0 - B)
    
    # 单位线UH1的卷积
    for k in range(max(1, min(NH - 1, int(Param[3] + 1)))):
        StUH1[k] = StUH1[k + 1] + OrdUH1[k] * PRUH1
    StUH1[NH - 1] = OrdUH1[NH - 1] * PRUH1
    
    # 单位线UH2的卷积
    for k in range(max(1, min(2 * NH - 1, 2 * int(Param[3] + 1)))):
        StUH2[k] = StUH2[k + 1] + OrdUH2[k] * PRUH2
    StUH2[2 * NH - 1] = OrdUH2[2 * NH - 1] * PRUH2
    
    # 潜在流域间半交换
    EXCH = Param[1] * (St[1] / Param[2] - Param[4])
    
    # 汇流库
    AEXCH1 = EXCH
    if (St[1] + (1 - C) * StUH1[0] + EXCH) < 0:
        AEXCH1 = -St[1] - (1 - C) * StUH1[0]
    St[1] = St[1] + (1 - C) * StUH1[0] + EXCH
    if St[1] < 0:
        St[1] = 0.0
    
    # 加速计算
    Rr = St[1] / Param[2]
    Rr = Rr * Rr
    Rr = Rr * Rr
    QR = St[1] * (1.0 - 1.0 / math.sqrt(math.sqrt(1.0 + Rr)))
    
    St[1] = St[1] - QR
    
    # 指数库更新
    St[2] = St[2] + C * StUH1[0] + EXCH
    AR = St[2] / Param[5]
    if AR > 33.0:
        AR = 33.0
    if AR < -33.0:
        AR = -33.0
    
    if AR > 7.0:
        QREXP = St[2] + Param[5] / math.exp(AR)
    elif AR < -7.0:
        QREXP = Param[5] * math.exp(AR)
    else:
        QREXP = Param[5] * math.log(math.exp(AR) + 1.0)
    
    St[2] = St[2] - QREXP
    
    # 直接支流的径流QD
    AEXCH2 = EXCH
    if (StUH2[0] + EXCH) < 0:
        AEXCH2 = -StUH2[0]
    QD = max(0.0, StUH2[0] + EXCH)
    
    # 总径流
    Q = QR + QD + QREXP
    if Q < 0:
        Q = 0.0
    
    # 变量存储
    MISC[0] = E                      # PE     潜在蒸散发
    MISC[1] = P1                     # Precip 降水
    MISC[2] = St[0]                  # Prod   产流库水位
    MISC[3] = PN                     # Pn     净降雨
    MISC[4] = PS                     # Ps     填充产流库的部分
    MISC[5] = AE                     # AE     实际蒸散发
    MISC[6] = PERC                   # PERC   渗透
    MISC[7] = PR                     # PR     PR=PN-PS+PERC
    MISC[8] = StUH1[0]              # Q9     UH1出流
    MISC[9] = StUH2[0]              # Q1     UH2出流
    MISC[10] = St[1]                 # Rout   汇流库水位
    MISC[11] = EXCH                  # EXCH   潜在流域间交换
    MISC[12] = AEXCH1                # AEXCH1 汇流库实际交换
    MISC[13] = AEXCH2                # AEXCH2 直接支流实际交换
    MISC[14] = AEXCH1 + AEXCH2 + EXCH # AEXCH 总实际交换
    MISC[15] = QR                    # QR     汇流库出流
    MISC[16] = QREXP                # QRExp  指数库出流
    MISC[17] = St[2]                 # Exp    指数库水位
    MISC[18] = QD                    # QD     UH2支流交换后出流
    MISC[19] = Q                     # Qsim   流域出口模拟出流
    
    return Q, MISC

@njit
def _pyrun_GR6J(
    inputs_data,
    # inputs_precip:  np.ndarray,
    # inputs_pe:      np.ndarray, 
    Param:          np.ndarray,
    NH:             int,
    NMISC:          int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GR6J模型的主运行函数，初始化GR6J，获取参数，在每个时间步调用MOD_GR6J子程序，并存储最终状态
    
    Parameters:
    -----------
    inputs_precip : np.ndarray
        输入的总降水时间序列 [mm/day]
    inputs_pe : np.ndarray
        输入的潜在蒸散发时间序列 [mm/day]
    Param : np.ndarray
        参数集合，包含6个参数:
        - Param[0]: 产流库容量 (X1 - PROD) [mm]
        - Param[1]: 流域间交换系数 (X2 - CES1) [mm/day]
        - Param[2]: 汇流库容量 (X3 - ROUT) [mm]
        - Param[3]: 单位线时间常数 (X4 - TB) [day]
        - Param[4]: 流域间交换阈值 (X5 - CES2) [-]
        - Param[5]: 指数库时间常数 (X6 - EXP) [day]
    State_Start : np.ndarray
        模型运行开始时的状态变量 (库容水位[mm]和单位线储量[mm])
    ind_Outputs : np.ndarray
        输出序列的索引
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Outputs : 输出序列
        State_end : 模型运行结束时的状态变量
    """
    inputs_precip = inputs_data.climatic_forcing[:, 0]
    inputs_pe     = inputs_data.climatic_forcing[:, 1]

    l_inputs = len(inputs_precip)
    
    # 初始化
    St = np.zeros(3)  # 3个主要状态变量
    StUH1 = np.zeros(NH)  # UH1状态
    StUH2 = np.zeros(2 * NH)  # UH2状态
    
    # # 使用StateStart初始化模型状态
    # St[0] = State_Start[0]  # 产流库水位
    # St[1] = State_Start[1]  # 汇流库水位
    # St[2] = State_Start[2]  # 指数库水位
    
    # # 初始化单位线状态
    # for i in range(NH):
    #     StUH1[i] = State_Start[7 + i]
    # for i in range(2 * NH):
    #     StUH2[i] = State_Start[7 + NH + i]
    
    # 计算单位线序列
    D = 2.5
    OrdUH1 = UH1_D(Param[3], D)
    OrdUH2 = UH2_D(Param[3], D)
    
    # 初始化输出
    Outputs = np.full((l_inputs, NMISC), -999.999)
    
    # 时间循环
    for k in range(l_inputs):
        P1 = inputs_precip[k]
        E = inputs_pe[k]
        
        # 单个时间步的模型运行
        Q, MISC = _MOD_GR6J(St, StUH1, StUH2, OrdUH1, OrdUH2, Param, P1, E, NH, NMISC)
        
        Outputs[k, :] = MISC
    
    # 模型运行结束时的状态
    # State_end = np.zeros(len(State_Start))
    # State_end[0] = St[0]
    # State_end[1] = St[1]
    # State_end[2] = St[2]
    
    # for k in range(NH):
    #     State_end[7 + k] = StUH1[k]
    # for k in range(2 * NH):
    #     State_end[7 + NH + k] = StUH2[k]
    
    return Outputs