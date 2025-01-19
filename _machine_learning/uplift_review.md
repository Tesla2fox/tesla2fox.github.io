---
title: "A simple review for Uplift Models"
collection: machine_learning
type: "Undergraduate course"
permalink: /machine_learning/uplift_survey
#venue: "Beijing"
date: 2025-01-19
---



## Uplift Model Taxmony

> Meta-Learner 

Meta-learner based. The basic idea of this line is to use the existing prediction methods to build the estimator for the users’ responses, which may be global (i.e., S-Learner) or divided by the treatment and control groups (i.e., T-Learner) . Based on this, different two-step learners can be designed by introducing various additional operations, such as **X-Learner (Which is used to handle the inbalance dataSet. The training process is very similar with Double machine learning )** , R-Learner (?), and DR-Learner (==?==), etc.

> Tree based

The basic idea of this line is to use a tree structure to gradually divide the entire user population into the sub-populations that are sensitive to each treatment. The key step is to directly model the uplift using different splitting criteria, such as based on various distribution divergences [25] and the expected responses [29, 38]. In addition, causal forest [3] obtained by integrating multiple trees is another representative method on this line, and several variants have been proposed.



>  Neural network based.

The basic idea of this line is to take advantage of neural networks to design more complex and flexible estimators for the user’s response, and most of them can be seen as improvements of the T-learner.

![Editing a markdown file for a talk](/images/WX20250119-194151@2x.png)



## Review Table 

| Name                                                         | My Thinking                                                  | Ref                                                          | Time |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| TARNet                                                       | Introduces a shared layer to mitigate the high variance problem caused by T-Learner. | Estimating individual treatment effect: generalization bounds and algorithms | 2017 |
| CFRNet (CounterFactual Regression Network)                   | Adds Integral Probability Metrics (IPM) mechanism to balance the data distribution between Treatment and Control groups by calculating their distance/similarity. | Estimating individual treatment effect: generalization bounds and algorithms | 2017 |
| DragonNet (because the dragon has three heads.)              | Adds propensity score estimator $$g(⋅)$$ to incorporate propensity score weighting into the loss function, mitigating selection bias. Cal the crossentropy with the treatment. | Adapting Neural Networks for the Estimation of Treatment Effects | 2019 |
| EUEN (Explicit Uplift Effect Network)                        | Directly learns $$τ$$, using $$μC+τ$$ to represent μT, and performs gradient descent with T group labels for loss. | Addressing Exposure Bias in Uplift Modeling for Large-scale Online Advertising | 2021 |
| DIPN(deep-isotonic-promotion-network)                        | Predicts monotonic increases with the treatment increasing   | A framework for massive scale personalized promotion         | 2021 |
| FlexTENet (flexible Treatment effect net )                   | 1. Ideas from multi-task learning (domain adaptation) 2.Merges **multiple shared layers** between T and C heads to alleviate the high variance issue. | On Inductive Biases for Heterogeneous Treatment Effect Estimation | 2021 |
| DESCN (Dual Entire Space Cross Network)                      | Addressing Treatment Bias and Sample Imbalance:  By combining ESN and X-network, DESCN can alleviate issues related to treatment bias and sample imbalance present in historical data. A shared network is used within DESCN to learn both propensity scores and control responses simultaneously, capturing comprehensive information about the user representation. | DESCN: Deep Entire Space Cross Networks for Individual Treatment Effect Estimation | 2022 |
| ESPN (Entire Space Pricing Network )                         | Assumes user sensitivity curves are S-shaped. sigmod function parameters Learning | From ATA                                                     | 2022 |
| CANDY                                                        |                                                              |                                                              |      |
| EFIN                                                         |                                                              |                                                              |      |
| DRNet                                                        |                                                              |                                                              |      |
| EFIN(Explicit Feature Interaction-aware Uplift Network)      | **Feature Encoding Module**: Encodes user, contextual, and treatment features. **Self-Interaction Module**: Models users' natural responses without treatment influence using self-attention mechanisms. **Treatment-Aware Interaction Module**: Captures the interaction between treatment features and other features to accurately model ITE. **Intervention Constraint Module**: Balances ITE distribution between control and treatment groups to ensure robustness in non-random marketing scenarios. | Explicit Feature Interaction-aware Uplift Network for Online Marketing | 2023 |
| ECUP(Entire Chain Uplift method with context-enhanced learning) | **Entire Chain-Enhanced Network**: This network leverages user sequential patterns to estimate the individual treatment effect (ITE) throughout the entire conversion chain. It models how treatments impact each task and integrates prior task information to enhance context awareness, thereby capturing the treatment's influence on different tasks. **Treatment-Enhanced Network**: This network refines treatment-aware features through a Treatment Awareness Unit that discerns correlations between treatments and personal features. It then balances initial and treatment-aware embeddings using a Treatment-Enhanced Gate with bit-level weights, allowing for flexible feature adjustments under various treatments. | Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing | 2024 |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |
|                                                              |                                                              |                                                              |      |



### Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing

![Editing a markdown file for a talk](/images/WX20250119-215753@2x.png)

![Editing a markdown file for a talk](/images/WX20250119-215818@2x.png)



