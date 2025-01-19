
import pandas as pd 
import numpy as np 
def get_cost_curve(amount, cost_real, gain_real, amount_real, gain, cost, verbose=False):
    cost_real_centered = cost_real - cost_real.mean()
    ref_amount_max = amount.max()
    ref_amount_min = amount.min()
    idx_real = pd.Series(np.arange(len(amount)), np.unique(amount_real)).reindex(amount_real).values
    mask_left = np.where(cost < cost[np.arange(len(cost)), idx_real][:, None], 1, np.nan)
    mask_right = np.where(cost > cost[np.arange(len(cost)), idx_real][:, None], 1, np.nan)
    mask_eq = amount_real[:, None] == amount
    cost_eq = cost[mask_eq][:, None]
    gain_eq = gain[mask_eq][:, None]
    k_left = np.nanmin(np.fillna((gain_eq - gain) / ((cost_eq - cost) * mask_left), np.infty), 1)
    k_right = np.nanmax(np.fillna((gain_eq - gain) / ((cost_eq - cost) * mask_right), -np.infty), 1)
    if verbose: print(len(k_right))
    possible = k_left > k_right
    if verbose: print((possible.sum()))
    k_all = np.concatenate([k_left[possible], k_right[possible]])
    gain_diff = np.concatenate([gain_real[possible], -gain_real[possible]])
    cost_diff = np.concatenate([cost_real_centered[possible], -cost_real_centered[possible]])

    k_cost_gain = np.asarray([k_all, cost_diff, gain_diff]).T

    k_cost_gain_not_inf = k_cost_gain[~np.isinf(k_cost_gain[:, 0])]
    k_cost_gain_not_inf = pd.DataFrame(k_cost_gain_not_inf) \
        .groupby(k_cost_gain_not_inf[:, 0]) \
        .sum() \
        .reset_index()[['index', 1, 2]] \
        .values
    order = np.argsort(k_cost_gain_not_inf[:, 0])[::-1]
    #     print('order', order.shape, order[:3])
    #     print('k_cost_gain_not_inf', k_cost_gain_not_inf.shape, k_cost_gain_not_inf[:3])
    k_cost_gain_not_inf = k_cost_gain_not_inf[order]
    #     print('k_cost_gain_not_inf', k_cost_gain_not_inf.shape, k_cost_gain_not_inf[:3])
    gain0 = gain_real[amount_real == ref_amount_min].sum()
    cost0 = cost_real_centered[amount_real == ref_amount_min].sum()
    gain_max = gain_real[amount_real == ref_amount_max].sum()
    cost_max = cost_real_centered[amount_real == ref_amount_max].sum()
    #     print('gain0', gain0.shape, gain0[:3])

    if verbose: print('gain0={},cost0={}, gain_max={}, cost_max={}'.format(gain0, cost0, gain_max, cost_max))
    cost_curve = pd.Series(
        np.concatenate([[0], np.cumsum(k_cost_gain_not_inf[:, 2])]),
        np.concatenate([[0], np.cumsum(k_cost_gain_not_inf[:, 1])])
    )
    cost_curve.loc[cost_real_centered[amount_real == amount.max()].sum() - cost0] = \
        gain_real[amount_real == amount.max()].sum() - gain0
    cost_curve.sort_index(inplace=True)
    cost_curve = cost_curve.loc[cost_curve.index < (cost_max - cost0)]
    cost_curve0 = cost_curve.copy()
    cost_curve /= gain_max - gain0
    cost_curve.index /= cost_max - cost0
    cost_curve.loc[1.0] = 1.
    cost_curve_area = (
            pd.Series(cost_curve.index).diff() * (cost_curve + cost_curve.shift().fillna(0)).values / 2).sum()
    return cost_curve, cost_curve_area, cost_curve0, possible.sum()