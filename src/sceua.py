import numpy as np

from numba import njit
from utils import relative_error, nash_sutcliffe_efficiency, kling_gupta_efficiency, _cal_mean, _get_min_along_axis0, _get_max_along_axis0

@njit
def get_random_int(nps, npg):
    """生成随机整数序列"""
    available = np.arange(2, npg)  # 生成数组
    selected = np.random.choice(available, nps - 1, replace=False)  # 无放回抽样
    lcs = np.empty(nps, dtype=np.int32)  # 预分配数组
    lcs[0] = 0
    lcs[1:] = np.sort(selected)  # 排序选中的数
    return lcs

@njit
def check_boundary(s, bl, bu):
    flag = False
    for i in range(s.shape[0]):
        if s[i] < bl[i] or s[i] > bu[i]:
            flag = True
            break
    return flag

# @njit
def cceua(fn, s, sf, bl, bu, x_obs, y_obs, fn_hm):
    """CCEUA算法实现"""
    nopt = s.shape[1]
    alpha = 1.0
    beta = 0.5

    # 提取最优和最差点
    sw = s[-1, :]
    fw = sf[-1]

    # 计算质心（排除最差点）
    # ce = np.mean(s[:-1, :], axis=0)
    ce = np.empty(nopt, dtype=np.float64)
    for j in range(nopt):
        ce[j] = _cal_mean(s[:-1, j])  # 逐列计算均值

    # 尝试反射点
    snew = ce + alpha * (ce - sw)

    # 边界检查
    if check_boundary(snew, bl, bu):
        snew = bl + np.random.rand(nopt) * (bu - bl)
    
    fnew = fn(x_obs, snew, y_obs, fn_hm)

    # 反射失败，尝试收缩
    if fnew > fw:
        snew = sw + beta * (ce - sw)
        fnew = fn(x_obs, snew, y_obs, fn_hm)
        
        # 收缩失败，生成随机点
        if fnew > fw:
            snew = bl + np.random.rand(nopt) * (bu - bl)
            fnew = fn(x_obs, snew, y_obs, fn_hm)
    
    return snew, fnew

# @njit
def sceua(bounds, max_iter, n_complex, n_params, x_obs, y_obs, fn_hm, cost_function):
    """SCEUA主算法"""
    bl = bounds[0, :]
    bu = bounds[1, :]
    param_range = bu - bl
    nopt = n_params
    npg = 2 * nopt + 1
    nps = nopt + 1
    nspl = npg
    npt = npg * n_complex

    # 初始化种群
    # x = np.zeros((npt, nopt))
    x = bl + np.random.rand(npt, nopt) * param_range
    # 计算初始损失
    # xf = np.array([cost_function(x_obs, x[i,:], y_obs, fn_hm) for i in range(npt)])
    xf = np.empty(npt, dtype=np.float64)
    for i in range(npt):
        xf[i] = cost_function(x_obs, x[i,:], y_obs, fn_hm)

    idx = np.argsort(xf)
    x = x[idx, :]
    xf = xf[idx]

    bestx = x[0, :].copy()
    bestf = xf[0]
    calibration_result = np.full(max_iter, np.nan)

    for tt in range(max_iter):
        for igs in range(n_complex):
            # 划分复合体
            # k2 = [i * n_complex + igs for i in range(npg)]
            k2 = np.empty(npg, dtype=np.int32)
            for i in range(npg):
                k2[i] = i * n_complex + igs
            cx = x[k2, :].copy()
            cf = xf[k2].copy()

            for _ in range(nspl):
                # 选择单纯形
                lcs = get_random_int(nps, npg)

                s = cx[lcs, :]
                sf = cf[lcs]

                snew, fnew = cceua(cost_function, s, sf, bl, bu, x_obs, y_obs, fn_hm)
                
                # 替换最差点
                s[-1, :] = snew
                sf[-1] = fnew
                cx[lcs, :] = s
                cf[lcs] = sf

                # 重新排序
                idx = np.argsort(cf)
                cx = cx[idx, :]
                cf = cf[idx]

            x[k2, :] = cx
            xf[k2] = cf

        # 洗牌排序
        idx = np.argsort(xf)
        x = x[idx, :]
        xf = xf[idx]
        bestx = x[0, :].copy()
        bestf = xf[0]

        # 收敛判断
        min_x = _get_min_along_axis0(x)
        max_x = _get_max_along_axis0(x)
        gnrng_values = (max_x - min_x) / param_range
        gnrng = np.exp(np.mean(np.log(gnrng_values + 1e-10)))  # 避免 log(0)
        if gnrng < 1e-4:
            break
        
        calibration_result[tt] = bestf
        if tt > 10 and abs(calibration_result[tt] - calibration_result[tt-10]) < 1e-6:
            break
    return bestx, calibration_result
