---
title: "A simple foundation stone for causal inference"
collection: machine_learning
type: "Undergraduate course"
permalink: /machine_learning/uplift_foundation_stone
#venue: "Beijing"
date: 2025-11-02
---



# 01 – Introduction to Causality

The chapter begins by motivating why causal inference matters in modern data science: while machine learning excels at prediction tasks, many real-world questions are of the form “What would happen if we changed X?” — i.e., causal or counterfactual questions. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html))

It introduces the key idea that **association is not the same as causation**. Even when a treatment and outcome are strongly correlated, we cannot infer a causal effect unless we rule out other factors that differ systematically between treated and untreated units. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html))

The chapter then presents the **potential outcomes framework** (also known as the Rubin causal model):

- Each unit (i) has $(T_i\in{0,1})$ indicating whether it receives the treatment. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html))
- It has two potential outcomes: $(Y_{1i})$ if treated, and $(Y_{0i})$ if not treated. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html))
- The **individual treatment effect** is $(Y_{1i} - Y_{0i})$. But since we only ever observe one of $(Y_{1i})$ or $(Y_{0i})$ for each unit, we cannot compute that directly. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html))
- We therefore focus on estimands like the **average treatment effect (ATE)**: ($\mathrm{ATE} = E[Y_1 - Y_0]$), or the **average treatment effect on the treated (ATT)**: $(E[Y_1 - Y_0 \mid T=1])$. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html))

Next, the text explains **bias** in causal estimation:
 $E[Y \mid T=1] - E[Y \mid T=0]
 = \underbrace{E[Y_1 - Y_0 \mid T=1]}*{\text{ATT}}+ \underbrace{E[Y_0 \mid T=1] - E[Y_0 \mid T=0]}*{\text{Bias}}$
 This shows that the simple difference in means between **<u>*treated and untreated combines the causal effect plus a bias term if treated and untreated differ in their counterfactual outcomes before treatment*</u>**. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html))





Finally, the chapter wraps up with **key ideas**:

- Understand that making treated and untreated groups comparable (so that bias vanishes) is central to causal inference. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html))



## 02 – Introduction to Causality

### Summary

This chapter focuses on **randomised experiments (or randomised controlled trials, RCTs)** as the “gold standard” for causal inference. It explains how randomisation eliminates bias by ensuring the treatment assignment is independent of the potential outcomes. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/02-Randomised-Experiments.html))

Key points:

- Because $((Y_0, Y_1) \perp T)$, the simple difference in observed means between the treated and control groups equals the average treatment effect (ATE). ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/02-Randomised-Experiments.html))
- The chapter gives a concrete example: a study comparing online vs face-to-face instruction in a school setting, where students are randomly assigned. This illustrates how randomisation can make treated and untreated groups comparable and thus support causal claims. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/02-Randomised-Experiments.html))
- It emphasises that while randomised experiments are ideal, they are often expensive, unethical or infeasible in many real-world contexts (e.g., you cannot randomly assign people to smoke or not, or randomly set minimum wages). ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/02-Randomised-Experiments.html))
- It introduces the notion of the **assignment mechanism** – understanding how units are assigned to treatment is critical to causal inference, even when randomisation is not possible. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/02-Randomised-Experiments.html))
- The chapter concludes with “Key Ideas” summarising that randomised experiments are the simplest way to uncover causal effects by making treatment and control groups comparable, but that in observational settings one must still think in terms of the “ideal experiment” and assignment mechanism. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/02-Randomised-Experiments.html))

------

### Suggested slide-ready take-away

- Randomised experiments remove bias by making treated vs control groups equivalent on all pre-treatment variables.
- With ((Y_0, Y_1) \perp T), the observed difference in means = ATE.
- In practice: aim to ask “If I could run the ideal experiment, what would I do?” because many real contexts cannot randomise.
- When randomisation isn’t feasible, focus on understanding the assignment mechanism.





# 3. Confounding Bias VS Collider Bias

## 1. Confounding Bias

**Definition:**
Confounding bias occurs when a variable (C) affects both the treatment (T) and the outcome (Y). This “common cause” creates a spurious association between (T) and (Y) if (C) is not controlled.

**Characteristics:**

* The bias arises from a common cause (C \to T) and (C \to Y).
* If (C) is not adjusted for, the observed effect $(E[Y \mid T=1] - E[Y \mid T=0])$ is biased.
* Can be reduced or removed by controlling for confounders, randomization, or using instrumental variables.

**Example:**
Smoking (T) and lung cancer (Y) may be confounded by age (C), because age affects both smoking habits and cancer risk.

---

## 2. Selection Bias (Collider Bias)

**Definition:**
Selection bias occurs when you condition on a **common effect (collider)** (S) that is influenced by both the treatment (T) and the outcome (Y). This conditioning can create a spurious association between (T) and (Y), even if none exists in the population.

**Characteristics:**

* Unlike confounding, the bias arises from a common effect$ (T \to S \leftarrow Y)$.
* Conditioning on or selecting based on (S) opens a previously blocked path, introducing bias.
* Controlling the wrong variable (a collider) can actually create bias instead of removing it.

**Example:**
Studying the effect of a drug (T) on recovery (Y) but only including hospitalized patients (S) in the analysis. Hospitalization is affected by both treatment and illness severity, which can distort the observed relationship.

*实验分析时，拆分的人群--- 选择不均值。 例如选择全部成熟期的用户，而不是选择首次命中成熟期的用户。*

---

## 3. Key Differences

| Feature     | Confounding Bias                          | Selection/Collider Bias                                      |
| ----------- | ----------------------------------------- | ------------------------------------------------------------ |
| Mechanism   | Common cause affects treatment & outcome  | Treatment and outcome both affect a common effect (collider) |
| Occurs when | Confounders not controlled                | Conditioning on a collider or its descendant                 |
| Bias effect | Spurious correlation from common cause    | Spurious correlation induced by conditioning/selection       |
| How to fix  | Adjust for confounders, randomize, use IV | Avoid conditioning on colliders; careful sampling            |

**Summary:**

* **Confounding bias**: “Fake” correlation due to a shared cause.
* **Selection/collider bias**: “Fake” correlation due to conditioning on a shared effect.



# 4. Variables 

### Key take‐aways

- **Include** variables that are strong predictors of the outcome (even if not confounders) — they help precision.
- **Avoid** including variables that strongly predict treatment but not outcome — these harm precision.
- **Never include** post‑treatment variables (mediators) or colliders/common effects — these introduce bias and break causal identification.
- Good controls vs bad controls: being a confounder is the classic criterion, but thinking about *outcome‐prediction* and *treatment‐prediction* also matters in practice.
- Graphical causal models (DAGs) serve as useful guides to figure out which variables are safe to control for and which are not.
- Even in randomized settings, the choice of controls matters for precision. In observational settings, choice also affects bias (via confounding, selection).
- In short: “Beyond confounders” means: controlling for confounders is necessary but not sufficient; we must also think about other types of controls and their consequences (for bias *and* variance).

------

If you like, I can also extract the **code examples**, **graphical diagrams**, and **detailed caveats** from the chapter (e.g., how to compute standard error effects of controls) so you can apply them in your own causal work. Would you like me to do that?

# 5. Propensity Score

### 1. What is the Propensity Score?

- The propensity score is defined as the probability of receiving the treatment given observed covariates:
  $e(X) = P(T = 1 \mid X)$ ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))
- The key insight: instead of controlling for the full vector (X) of covariates to achieve
  $(Y_1, Y_0) \perp T \mid X$
   it is **sufficient** to condition on the propensity score:
   $(Y_1, Y_0) \perp T \mid e(X)$ ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))
- Intuitively, if two units have the same (e(X)), then which one receives treatment is (under the assumptions) “as good as random”—so balancing on the propensity score is enough to remove bias from observed confounders. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))

------

### 2. How to Use the Propensity Score

The webpage outlines several major applications:

- **Weighting (Inverse Probability of Treatment Weighting – IPTW):**
  - One can estimate the ATE by weighting each unit by the inverse of the probability of the treatment they actually received:
     $w_i = \frac{T_i}{e(X_i)} + \frac{1 - T_i}{1 - e(X_i)}$
     Then compute a weighted average of the outcome. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))
  - Requires the “positivity” or “overlap” assumption: no one has zero or one probability of treatment (i.e., (e(X)) strictly between 0 and 1) so weights don’t explode. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))
- **Matching on the Propensity Score:**
  - Instead of matching on full (X), match treated and untreated units with similar (e(X)). This is a dimension‑reduction trick to ease matching when (X) is high‑dimensional. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))
- **Stratification / subclassification on the Propensity Score:**
  - Divide sample into bins (e.g., quintiles) by (e(X)), and compare treated vs untreated within each bin, then aggregate. (Mentioned implicitly in the method list).
- **Using (e(X)) in Regression:**
  - The propensity score may be included as a covariate or used in regression adjustment (or even both) to adjust for confounding. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))

------

### 3. Practical Estimation and Diagnostics

- In practice, the true $(e(X))$ is unknown; we estimate $(\hat e(X))$ typically via logistic regression (or other machine‑learning tools) using observed (X). ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))
- After estimating $(\hat e(X))$, diagnostics are essential:
  - **Balance check:** Examine whether treated and control units are similar in (X) within strata or weights of (e(X)). ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))
  - **Overlap check (positivity):** Check distributions of ($\hat e(X)$) for treated and untreated; if there’s little to no overlap (e.g., treated always high (e(X)) and untreated always low), then causal estimation is weak. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))
- Estimation of standard errors: when using IPTW, standard errors based on simple weighted averages assume known (e(X)); if (\hat e(X)) is estimated, one should use bootstrapping (or other variance‑adjusted methods) to account for propensity score estimation error. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))

------

### 4. Common Issues and Pitfalls

- **Predictive accuracy ≠ balancing quality:** A model for (e(X)) that predicts (T) extremely well (very high AUC) is not necessarily good for causal balance. In fact, including variables that strongly predict treatment but are unrelated to outcome can increase variance (or even bias) of effect estimates. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))
- **Poor overlap (positivity violation):** If some units have extremely low or high propensity scores (near 0 or 1), weights become extreme and variance inflates; also, matching/extrapolation becomes risky. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))
- **Weights extreme/clipping:** Often practitioners cap (clip) weights (for example maximum weight = 20) to limit variance, but this introduces additional bias and should be used carefully. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))
- **Standard error under‑estimation:** Many simple implementations ignore the estimation error in (\hat e(X)), so standard errors may be under­estimated. ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html))

------

### 5. Key Take‑aways

- The propensity score is a powerful tool for reducing bias from observable confounders by summarizing high‐dimensional (X) into a single score (e(X)).
- It enables methods like weighting, matching, stratification, and regression adjustment, often improving design phase of causal inference (before outcome modelling).
- However, one must carefully assess its assumptions: overlap/positivity, correct model estimation, good balance and non‑extreme weights. The method does **not** address unobserved confounders.
- Using propensity score methods effectively is as much about diagnostics (balance, overlap) and design as it is about modelling.



# 6. Doubly Robust Estimation

### 1. What is Doubly Robust Estimation?

- When estimating a causal effect in observational data, you often rely on two kinds of models:
  1. A model for the **propensity score** (the probability of treatment given covariates)
  2. A model for the **outcome regression** (expected outcome given treatment and covariates)
- The doubly robust (DR) estimator **combines** these two models in one formula so that it remains **consistent (unbiased in large samples)** if *either one* of the two models is correctly specified (though not necessarily both). ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/12-Doubly-Robust-Estimation.html?utm_source=chatgpt.com))
- In formula form (simplified):
  $\hat{ATE} = \frac{1}{N} \sum_{i=1}^N \Big( \frac{T_i (Y_i - \hat\mu_1(X_i))}{\hat P(X_i)} + \hat\mu_1(X_i) \Big) ;-; \frac{1}{N} \sum_{i=1}^N \Big( \frac{(1-T_i)(Y_i - \hat\mu_0(X_i))}{1 - \hat P(X_i)} + \hat\mu_0(X_i)\Big)$
   where:
  - $\hat P(X_i)$ = estimated propensity score
  - $\hat\mu_1(X_i) = \hat E(Y|T=1, X_i)$ = estimated outcome model under treatment
  - $\hat\mu_0(X_i) = \hat E(Y|T=0, X_i)$ = estimated outcome model under control ([matheusfacure.github.io](https://matheusfacure.github.io/python-causality-handbook/12-Doubly-Robust-Estimation.html?utm_source=chatgpt.com))
- The name “doubly robust” comes from the “either/or” nature: as long as **either** the propensity model **or** the outcome regression model is correct, the estimator converges to the true effect.

------

### 2. Why use it? What are the advantages?

- If you use just an outcome regression model and you misspecify it, you’ll get bias (because of residual confounding / model misfit).
- If you use just a propensity score weighting model (IPTW) and you misspecify it, you’ll also get bias (or large variance) because the weights may be wrong or extreme.
- The DR estimator **guards against** one of these model mis‐specifications — you only need one of them to be correctly specified to still get a consistent estimate.
- It thus offers a kind of “safety net” when you’re uncertain whether your specification for the propensity model or the outcome model is fully correct.

------

### 3. Things to watch out for / limitations

- Although DR is more robust than single‐model methods, **if both models (propensity and outcome) are mis‐specified**, the estimator can still be biased. As one review article said: “DR produces an unbiased estimator unless both PS and outcome models are incorrectly specified.” ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8098857/?utm_source=chatgpt.com))
- Even if one model is correct, there are practical issues: extreme weights (from propensity score near 0 or 1), poor overlap, limited sample size, or highly complex outcome relationships can degrade performance.
- Model specification still matters: if your outcome regression is badly wrong and your propensity model is also badly wrong, you lose the benefit.
- Standard errors / inference: sometimes extra care is needed to account for estimation of the nuisance models (propensity and outcome).
- The estimator still relies on the usual causal inference assumptions: no unmeasured confounding (given (X)), overlap/positivity, correct specification of at least one model, etc.

------

### 4. Practical usage / intuition

- In practice you would:
  1. Estimate the propensity score model $\hat P(X) = P(T=1|X)$ ) (e.g., logistic regression)
  2. Estimate the outcome regression models for treated and control: $\hat\mu_1(X) \approx E[Y|T=1, X] $) and $ \hat\mu_0(X) \approx E[Y|T=0, X] $ (e.g., linear regression, machine learning)
  3. Plug into the DR formula above to compute the ATE (or whichever estimand you care about)
  4. Check diagnostics: balance on (X), overlap of propensity scores, distribution of weights, sensitivity if one model is plausibly wrong
- The intuition: You are using two “paths” to adjust for confounding — a **model for selection into treatment** + a **model for the outcome given covariates** — and if at least one path is credible, you get a good estimate.

------

### 5. Key take‐aways

- Doubly robust estimation is a valuable technique in observational causal inference because it **reduces reliance** on having *both* models perfectly specified.
- It **combines** propensity score modeling and outcome modeling in one unified estimator.
- Its consistency requires that **at least one** of the two models is correct — hence “double robustness.”
- But it is *not* a magical cure: if both models fail badly, you'll get bias. It also doesn’t substitute for sound causal assumptions (e.g., unmeasured confounding is still a problem).
- Use in practice: when you have observational data, estimate both models, use DR as part of your toolkit, check diagnostics, and interpret results carefully.



