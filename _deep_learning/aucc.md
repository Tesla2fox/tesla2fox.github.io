---
title: "Metrics for Uplift Models"
collection: deep_learning
type: "Undergraduate course"
permalink: /deep_learning/pawa_test
#venue: "Beijing"
date: 2024-12-29
---


Paliang's log
======

## Inverse Probability Weighted Estimator (IPWE)

The Inverse Probability Weighted Estimator (IPWE) is a key method used in causal inference to address potential confounding in observational studies, especially when treatment assignment is not entirely random. By using IPWE, we aim to unbiasedly estimate the potential outcomes under different interventions, such as costs and gains.



**Inverse probability weighting recovers consistent estimates when data are missing at random.**



![Editing a markdown file for a talk](/images/WX20241229-212819@2x.png)

## Area Under the Uplift Curve (AUUC)

In uplift research, to evaluate the rankability of the uplift model $$y$$, one can first plot an uplift curve which ranks individual samples descendingly according to ==𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑒𝑑 uplift value==  $$\hat{\tau}$$ (in X-axis) and cumulatively sums ==the 𝑜𝑏𝑠𝑒𝑟𝑣𝑒𝑑 uplift value== (in Y-axis). The AUUC is then the area under this curve. There are actually multiple variants of uplift curves proposed in the recent literature. Their differences mainly lie in 1) if ranking the data separately per group or jointly over all data, and 2) if expressing volumes in absolute or relative numbers.



First, we denote the total number of treated and control instances, among the top-𝑘 individuals $${\pi}(D,𝑘)$$ ranked by uplift model $$𝑓$$ over the whole dataset $$D$$ as


$$N_{\pi}^{T}(D, k) = \sum_{(x_i, t_i, y_i) \in \pi(D, k)} I(t_i = 1)$$

$$N_{\pi}^{C}(D, k) = \sum_{(x_i, t_i, y_i) \in \pi(D, k)} I(t_i = 0)$$

$$N_{\pi}^{T}(D, k)$$ means the numbe of the treatment set, $$N_{\pi}^{C}(D, k)$$ means the numbe of  the control set. 

-------

$$R^T_{\pi}(D,k) = \sum_{(x_i,t_i,y_i) \in \pi(D,k)} y_i I(t_i = 1)$$

$$R^C_{\pi}(D,k) = \sum_{(x_i,t_i,y_i) \in \pi(D,k)} y_i I(t_i = 0)$$

$$R_{\pi}^{T}(D, k)$$ means the total response of treatment set, $$R_{\pi}^{C}(D, k)$$ means the total repsonse of the control set. 

-------------

The each point value for the uplift curve can be obtained with

$$V_u(f,k) = \left( \frac{R^T_{\pi}(D,k)}{N^T_{\pi}(D,k)} - \frac{R^C_{\pi}(D,k)}{N^C_{\pi}(D,k)} \right) \times (N^T_{\pi}(D,k) + N^C_{\pi}(D,k))$$



Finally, 

$$AUUC(f) = \int_0^1 V_u(f, x) \, dx = \frac{1}{n} \sum_{k=1}^{n} V_u(f, k) \approx \sum_{p=1}^{100} V_u\left(f, \frac{p}{100}\right)$$

**remark** 

-  In this definition of AUUC, the value is not formulated to 0-1.   In the real application, we need to ==normalize== the AUUC.
-  $$\frac{N^C_{\pi}(D,k)}{N^T_{\pi}(D,k) + N^C_{\pi}(D,k)}$$ and $$\frac{N^T_{\pi}(D,k)}{N^T_{\pi}(D,k) + N^C_{\pi}(D,k)}$$ can be seen the Inverse Probability Weighted (IPWE) of control group and treatment group.

## Area Under the Cost Curve (AUCC)

AUUC doesn't consider the cost of a treatment. However, in industrial, a treatment must be with a cost (such as a coupon or an exposure of video). Maybe there are different costs between a and b. When the treatments on different things have different costs, we need to use AUCC to replace AUUC to measure uplift model $$f$$ preformance.



## Area Under the QINI Curve (AUQC):

The each point value for the ==QINI== curve is shown as follows.

$$V_u(f,k) = R^T_{\pi}(D,k) - \frac{N^T_{\pi}(D,k) }{N^C_{\pi}(D,k)} \times R^C_{\pi}(D,k) $$



## Kendall Rank Correlation Coefficient (KRCC): 

The KRCC measures the similarity between the rank by predicted uplift scores and the rank by the approximated true uplift scores for all individuals. 

To calculate the Kendall Rank Correlation Coefficient (Kendall's tau), you can follow these steps. This coefficient measures the ordinal association between two measured quantities and is particularly useful for non-parametric data. Here’s a detailed guide on how to compute it:

### Steps to Calculate Kendall's Tau

1. **Prepare Your Data**:
   - Assume you have two variables \\(X\\) and \\(Y\\), each with \\(n\\) observations, denoted as \\((x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\\).

2. **Consider All Possible Pairs**:
   - Look at all possible pairs of observations \((i, j)\) where \(1 \leq i < j \leq n\). There will be \(\frac{n(n-1)}{2}\) such pairs.

3. **Define Concordant and Discordant Pairs**:
   - A pair \((x_i, y_i)\) and \((x_j, y_j)\) is considered concordant if the ranks for both elements agree; that is, if \(x_i < x_j\) and \(y_i < y_j\) or \(x_i > x_j\) and \(y_i > y_j\).
   - The pair is discordant if the ranks disagree; that is, if \(x_i < x_j\) and \(y_i > y_j\) or \(x_i > x_j\) and \(y_i < y_j\).
   - If \(x_i = x_j\) or \(y_i = y_j\), the pair is neither concordant nor discordant and is called a tie.

4. **Count Concordant and Discordant Pairs**:
   - Let \(C\) be the number of concordant pairs.
   - Let \(D\) be the number of discordant pairs.

5. **Calculate Kendall's Tau**:
   - The basic formula for Kendall's tau is:
     $$ \tau = \frac{C - D}{\frac{1}{2} n (n-1)} $$
   - When there are ties in the data, an adjusted version known as Kendall's tau-b should be used:
     \[
     \tau_b = \frac{C - D}{\sqrt{(C + D + T_x)(C + D + T_y)}}
     \]
     Where \(T_x\) and \(T_y\) represent the number of ties involving only the \(X\) variable and only the \(Y\) variable, respectively. For each set of tied ranks, the count is calculated as the sum of \(\binom{s}{2} = \frac{s(s-1)}{2}\) for all groups of ties of size \(s\).

6. **Interpret the Result**:
   - The value of Kendall's tau ranges from -1 to +1.
   - A value of +1 indicates perfect agreement in ranking.
   - A value of -1 indicates perfect disagreement in ranking.
   - A value near 0 suggests no relationship between rankings.



## LIFT@h

This metric measures the difference between the mean response of treated individuals and that of controlled individuals in top $$h$$ percentile of all individuals ranked by the uplift model.

## All those mentioned metrics is designed for a single-treatment uplift model. How to evaluate the model which is designed for multi-treatment scene?







    cost_real_centered = cost_real - cost_real.mean()
    ref_amount_max = amount.max()
    ref_amount_min = amount.min()
    idx_real = pd.Series(np.arange(len(amount)), np.unique(amount_real)).reindex(amount_real).values
    mask_left = np.where(cost < cost[np.arange(len(cost)), idx_real][:, None], 1, np.nan)
    mask_right = np.where(cost > cost[np.arange(len(cost)), idx_real][:, None], 1, np.nan)
    mask_eq = amount_real[:, None] == amount
    cost_eq = cost[mask_eq][:, None]
    gain_eq = gain[mask_eq][:, None]
    k_left = np.nanmin(fillna((gain_eq - gain) / ((cost_eq - cost) * mask_left), np.infty), 1)
    k_right = np.nanmax(fillna((gain_eq - gain) / ((cost_eq - cost) * mask_right), -np.infty), 1)
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