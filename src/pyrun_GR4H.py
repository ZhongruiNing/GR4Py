#------------------------------------------------------------------------------
#    Subroutines relative to the annual GR4H model
#------------------------------------------------------------------------------
# TITLE   : GR4Py
# PROJECT : GR4Py
# FILE    : pyrun_GR4H.py
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
#         1. pyrun_GR4H
#         2. MOD_GR4H
#------------------------------------------------------------------------------

import numpy as np

from typing import Tuple
from numba import njit
from utils import UH1_H, UH2_H


@njit
def _MOD_GR4H(St:       np.ndarray, 
              StUH1:    np.ndarray, 
              StUH2:    np.ndarray, 
              OrdUH1:   np.ndarray, 
              OrdUH2:   np.ndarray, 
              Param:    np.ndarray, 
              P1:       float, 
              E:        float,
              NH:       int,
              NMISC:    int
              ) -> Tuple[float, np.ndarray]:
    """
    使用 GR4H 模型计算单个时间步（小时）的流量。

    参数说明:
    --------
    St : np.ndarray
        时间步开始时各库的模型状态 [mm]
    StUH1 : np.ndarray
        时间步开始时单位线水库1的模型状态 [mm]
    StUH2 : np.ndarray
        时间步开始时单位线水库2的模型状态 [mm]
    OrdUH1 : np.ndarray
        UH1 的各阶值 [-]
    OrdUH2 : np.ndarray
        UH2 的各阶值 [-]
    Param : np.ndarray
        模型参数 [单位各异]
    P1 : float
        时间步内的降水量 [mm]
    E : float
        时间步内的潜在蒸散量 [mm]

    返回值:
    -------
    Tuple[float, np.ndarray]
        Q: 该时间步流域出口的模拟流量 [mm/小时]
        MISC: 该时间步的模型输出 [mm/小时]
    """
    
    # Constants
    B = 0.9
    stored_val = 759.69140625
    
    # Initialize variables
    MISC = np.full(NMISC, -999.999, dtype=np.float64)
    
    A = Param[0]
    
    # Interception and production store
    if P1 <= E:
        EN = E - P1
        PN = 0.0
        WS = EN / A
        if WS > 13.0:
            WS = 13.0
        
        # Speed-up calculation
        expWS = np.exp(2.0 * WS)
        TWS = (expWS - 1.0) / (expWS + 1.0)
        Sr = St[0] / A
        ER = St[0] * (2.0 - Sr) * TWS / (1.0 + (1.0 - Sr) * TWS)
        
        AE = ER + P1
        St[0] = St[0] - ER
        PR = 0.0
        PS = 0.0
    else:
        EN = 0.0
        AE = E
        PN = P1 - E
        WS = PN / A
        if WS > 13.0:
            WS = 13.0
        
        # Speed-up calculation
        expWS = np.exp(2.0 * WS)
        TWS = (expWS - 1.0) / (expWS + 1.0)
        Sr = St[0] / A
        PS = A * (1.0 - Sr * Sr) * TWS / (1.0 + Sr * TWS)
        
        PR = PN - PS
        St[0] = St[0] + PS
    
    # Percolation from production store
    if St[0] < 0.0:
        St[0] = 0.0
    
    # Speed-up calculation for percolation
    Sr = St[0] / Param[0]
    Sr = Sr * Sr
    Sr = Sr * Sr
    PERC = St[0] * (1.0 - 1.0 / np.sqrt(np.sqrt(1.0 + Sr / stored_val)))
    
    St[0] = St[0] - PERC
    PR = PR + PERC
    
    # Split of effective rainfall into the two routing components
    PRHU1 = PR * B
    PRHU2 = PR * (1.0 - B)
    
    # Convolution of unit hydrograph UH1
    max_k1 = max(1, min(NH - 1, int(Param[3] + 1.0)))
    for K in range(max_k1):
        StUH1[K] = StUH1[K + 1] + OrdUH1[K] * PRHU1
    StUH1[NH - 1] = OrdUH1[NH - 1] * PRHU1
    
    # Convolution of unit hydrograph UH2
    max_k2 = max(1, min(2 * NH - 1, 2 * int(Param[3] + 1.0)))
    for K in range(max_k2):
        StUH2[K] = StUH2[K + 1] + OrdUH2[K] * PRHU2
    StUH2[2 * NH - 1] = OrdUH2[2 * NH - 1] * PRHU2
    
    # Potential intercatchment semi-exchange
    # Speed-up calculation
    Rr = St[1] / Param[2]
    EXCH = Param[1] * Rr * Rr * Rr * np.sqrt(Rr)
    
    # Routing store
    AEXCH1 = EXCH
    if (St[1] + StUH1[0] + EXCH) < 0.0:
        AEXCH1 = -St[1] - StUH1[0]
    St[1] = St[1] + StUH1[0] + EXCH
    if St[1] < 0.0:
        St[1] = 0.0
    
    # Speed-up calculation for routing
    Rr = St[1] / Param[2]
    Rr = Rr * Rr
    Rr = Rr * Rr
    QR = St[1] * (1.0 - 1.0 / np.sqrt(np.sqrt(1.0 + Rr)))
    
    St[1] = St[1] - QR
    
    # Runoff from direct branch QD
    AEXCH2 = EXCH
    if (StUH2[0] + EXCH) < 0.0:
        AEXCH2 = -StUH2[0]
    QD = max(0.0, StUH2[0] + EXCH)
    
    # Total runoff
    Q = QR + QD
    if Q < 0.0:
        Q = 0.0
    
    # 变量存储
    MISC[0] = E              # PE     - 观测的潜在蒸散量 [mm/小时]
    MISC[1] = P1             # Precip - 观测的总降水量 [mm/小时]
    MISC[2] = St[0]          # Prod   - 产水库容水量 (St[0]) [mm]
    MISC[3] = PN             # Pn     - 净降水量 [mm/小时]
    MISC[4] = PS             # Ps     - 填充产水库的部分降水量 [mm/小时]
    MISC[5] = AE             # AE     - 实际蒸散量 [mm/小时]
    MISC[6] = PERC           # Perc   - 渗漏量 (PERC) [mm/小时]
    MISC[7] = PR             # PR     - PR=PN-PS+PERC [mm/小时]
    MISC[8] = StUH1[0]       # Q9     - UH1单元水库的流出量 (Q9) [mm/小时]
    MISC[9] = StUH2[0]       # Q1     - UH2单元水库的流出量 (Q1) [mm/小时]
    MISC[10] = St[1]         # Rout   - 汇流库容水量 (St[1]) [mm]
    MISC[11] = EXCH          # Exch   - 潜在汇流区间交换量 (EXCH) [mm/小时]
    MISC[12] = AEXCH1        # AExch1 - 汇流库实际交换量 (AEXCH1) [mm/小时]
    MISC[13] = AEXCH2        # AExch2 - 直接分支（UH2后）实际交换量 (AEXCH2) [mm/小时]
    MISC[14] = AEXCH1 + AEXCH2  # AExch  - 总实际交换量 (AEXCH1+AEXCH2) [mm/小时]
    MISC[15] = QR            # QR     - 汇流库流出量 (QR) [mm/小时]
    MISC[16] = QD            # QD     - 交换后UH2分支流出量 (QD) [mm/小时]
    MISC[17] = Q             # Qsim   - 流域出口模拟流量 [mm/小时]
    
    return Q, MISC


@njit
def _pyrun_GR4H(inputs_data,
               Param: np.ndarray, 
               NH:             int,
               NMISC:          int,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    frun_gr4h 子程序的 Python 版本，初始化 GR4H，
    获取其参数，在每个时间步调用 MOD_GR4H 子程序，
    并存储最终状态。

    参数说明:
    --------
    l_inputs : int
        输入和输出序列的长度
    inputs_precip : np.ndarray
        总降水量输入序列 [mm/小时]
    inputs_pe : np.ndarray
        潜在蒸散量 (PE) 输入序列 [mm/小时]
    NParam : int
        模型参数数量
    Param : np.ndarray
        参数集
    NStates : int
        状态变量数量
    StateStart : np.ndarray
        模型运行开始时使用的状态变量（库容水量 [mm] 和单位线水库 (UH) 储水量 [mm]）
    NOutputs : int
        输出序列数量
    IndOutputs : np.ndarray
        输出序列的索引

    返回值:
    -------
    Tuple[np.ndarray, np.ndarray]
        Outputs: 输出序列
        StateEnd: 模型运行结束时的状态变量
    """

    inputs_precip = inputs_data.climatic_forcing[:, 0]
    inputs_pe     = inputs_data.climatic_forcing[:, 1]

    l_inputs  = len(inputs_precip)

    # Initialize states
    St = np.zeros(2, dtype=np.float64)
    StUH1 = np.zeros(NH, dtype=np.float64)
    StUH2 = np.zeros(2 * NH, dtype=np.float64)
    
    # # Initialize model states using StateStart
    # St[0] = StateStart[0]  # Python 0-based indexing
    # St[1] = StateStart[1]
    
    # # Initialize UH storages
    # for i in range(NH):
    #     StUH1[i] = StateStart[6 + i]  # 7+i in Fortran becomes 6+i in Python
    
    # for i in range(2 * NH):
    #     StUH2[i] = StateStart[6 + NH + i]  # 7+NH+i in Fortran becomes 6+NH+i in Python
    
    # Parameter values:
    # Param[0]: production store capacity (X1 - PROD) [mm]
    # Param[1]: intercatchment exchange coefficient (X2 - CES) [mm/hour]
    # Param[2]: routing store capacity (X3 - ROUT) [mm]
    # Param[3]: time constant of unit hydrograph (X4 - TB) [hour]
    
    # Computation of UH ordinates
    D = 1.25
    OrdUH1 = UH1_H(Param[3], D)
    OrdUH2 = UH2_H(Param[3], D)
    
    # Initialize outputs
    Outputs = np.full((l_inputs, NMISC), -999.999)
    
    # Time loop
    for k in range(l_inputs):
        P1 = inputs_precip[k]
        E = inputs_pe[k]
        
        # Model run on one time step
        Q, MISC = _MOD_GR4H(St, StUH1, StUH2, OrdUH1, OrdUH2, Param, P1, E, NH, NMISC)
        
        # Storage of outputs
        Outputs[k, :] = MISC
    
    # Model states at the end of the run
    # StateEnd = np.zeros(NStates, dtype=np.float64)
    # StateEnd[0] = St[0]
    # StateEnd[1] = St[1]
    
    # for k in range(NH):
    #     StateEnd[6 + k] = StUH1[k]  # 7+k in Fortran becomes 6+k in Python
    
    # for k in range(2 * NH):
    #     StateEnd[6 + NH + k] = StUH2[k]  # 7+NH+k in Fortran becomes 6+NH+k in Python
    
    return Outputs
