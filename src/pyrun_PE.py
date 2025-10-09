#------------------------------------------------------------------------------
#    Subroutines relative to the Oudin potential evapotranspiration (PE) formula
#------------------------------------------------------------------------------
# TITLE   : GR4Py
# PROJECT : GR4Py
# FILE    : pyrun_PE.py
#------------------------------------------------------------------------------
# AUTHORS
# Original code: Zhongrui Ning
#------------------------------------------------------------------------------
# Creation date: 2025
# Last modified: 09/19/2025
#------------------------------------------------------------------------------
# REFERENCES
# Oudin, L., Hervieu, F., Michel, C., Perrin, C., Andréassian, V.,
# Anctil, F. and Loumagne, C. (2005). Which potential evapotranspiration
# input for a lumped rainfall-runoff model? Part 2 - Towards a simple and
# efficient potential evapotranspiration model for rainfall-runoff modelling.
# Journal of Hydrology, 303(1-4), 290-306, doi: 10.1016/j.jhydrol.2004.08.026.
#------------------------------------------------------------------------------
# Quick description of public procedures:
#         1. pyrun_PE_OUDIN
#         2. PE_OUDIN
#------------------------------------------------------------------------------

import numpy as np
import math

def PE_OUDIN(FI, DT, JD):
    """
    Calculate potential evapotranspiration (PE) for a single time step
    using air temperature and daily extra-atmospheric global radiation
    
    Parameters:
    -----------
    FI : float
        Latitude [rad]
    DT : float
        Air Temperature [°C]
    JD : float
        Julian day [-]
    
    Returns:
    --------
    DPE : float
        Potential evapotranspiration [mm/time step]
    """
    # Calculation of extra-atmospheric global radiation
    # (Appendix C in Morton (1983), Eq. C-6 to C-11, p.60-61)
    COSFI = math.cos(FI)
    
    # TETA: Declination of the sun in radians
    TETA = 0.4093 * math.sin(JD / 58.1 - 1.405)
    COSTETA = math.cos(TETA)
    COSGZ = max(0.001, math.cos(FI - TETA))
    
    # GZ: Noon angular zenith distance of the sun
    GZ = math.acos(COSGZ)
    COSGZ2 = COSGZ * COSGZ
    
    if COSGZ2 >= 1.0:
        SINGZ = 0.0
    else:
        SINGZ = math.sqrt(1.0 - COSGZ2)

    COSOM = 1.0 - COSGZ / COSFI / COSTETA
    COSOM = max(-1.0, min(1.0, COSOM))  # Clamp between -1 and 1
    COSOM2 = COSOM * COSOM
    
    if COSOM2 >= 1.0:
        SINOM = 0.0
    else:
        SINOM = math.sqrt(1.0 - COSOM2)
    
    OM = math.acos(COSOM)
    
    # PZ: Average angular zenith distance of the sun
    COSPZ = COSGZ + COSFI * COSTETA * (SINOM / OM - 1.0)
    COSPZ = max(0.001, COSPZ)
    
    # ETA: Radius vector of the sun
    ETA = 1.0 + math.cos(JD / 58.1) / 30.0
    
    # GE: extra-atmospheric global radiation
    GE = 446.0 * OM * COSPZ * ETA
    
    # Daily PE by Oudin et al. (2005) formula
    DPE = max(0.0, GE * (DT + 5.0) / 100.0 / 28.5)

    return DPE


def pyrun_PE_OUDIN(inputs_lat, inputs_temp, inputs_jj):
    """
    Perform PE calculation for entire time series using Oudin formula
    
    Parameters:
    -----------
    inputs_lat : array_like
        Input series of latitude [rad]
    inputs_temp : array_like
        Input series of air mean temperature [°C]
    inputs_jj : array_like
        Input series of Julian day [-]
    
    Returns:
    --------
    outputs_pe : numpy.ndarray
        Output series of potential evapotranspiration [mm/time step]
    """
    # Convert inputs to numpy arrays
    inputs_lat = np.asarray(inputs_lat)
    inputs_temp = np.asarray(inputs_temp)
    inputs_jj = np.asarray(inputs_jj)
    
    # Check that all inputs have the same length
    l_inputs = len(inputs_lat)
    
    # Initialize output array
    outputs_pe = np.zeros(l_inputs)
    
    # Time loop
    for k in range(l_inputs):
        TT = inputs_temp[k]
        JJ = inputs_jj[k]
        FI = inputs_lat[k]
        
        # Model run on one time step
        pe_oud = PE_OUDIN(FI, TT, JJ)
        
        # Storage of outputs
        outputs_pe[k] = pe_oud
    
    return outputs_pe