---
title: "Some Questions about logistic regression"
collection: machine_learning
type: "Undergraduate course"
permalink: /machine_learning/logistic_regression
#venue: "Beijing"
date: 2025-01-19
---



# LR

## **1. Basic Theory Questions**

### Core Concepts:

1. What is logistic regression? How does it work?

**Answer:** Logistic regression is a supervised machine learning algorithm used for binary classification problems. It models the relationship between a dependent binary variable and one or more independent variables by using a logistic (Sigmoid) function.

The logistic function, $$σ(z)=\frac{1}{1 + e^{-z}}σ(z)$$, maps the linear combination of input features (weighted sum) to a probability value between 0 and 1. The output probability represents the likelihood of the instance belonging to the positive class (usually labeled as 1).

2. Why is logistic regression a classification algorithm and not a regression algorithm?

**Answer:** Logistic regression is called "regression" due to its use of a linear model (weighted sum of features) to predict outcomes. However, it’s primarily used for classification because it transforms the linear output through a **Sigmoid function**, which converts the result into a probability score between 0 and 1. This is interpreted as a probability of class membership, not a continuous value.

3. How does logistic regression differ from linear regression?

**Answer:**

- **Linear regression** predicts a continuous output based on a linear relationship between input features and output.
- **Logistic regression** is used for classification tasks, predicting the probability of a binary outcome, and applies a Sigmoid function to map the linear output to a value between 0 and 1.

Linear regression minimizes mean squared error (MSE), while logistic regression minimizes log-loss (cross-entropy loss).

4. What is the role of the Sigmoid function in logistic regression?

**Answer:** The Sigmoid function maps any real-valued number into the range (0, 1). This is useful for binary classification, as the output can be interpreted as the probability of the instance belonging to the positive class. The function is defined as:

$$σ(z)=\frac{1}{1 + e^{-z}}σ(z)$$

where zzz is the linear combination of the features.

5. Explain how logistic regression models probabilities.



6. What are some assumptions made by logistic regression?

**Answer:**

- **Linearity**: Assumes a linear relationship between the input variables and the log-odds of the outcome.
- **Independence**: Assumes that the observations are independent of each other.
- **No multicollinearity**: Assumes no perfect correlation between the independent variables.

补充>

变量得是分类变量，训练的样本数量有要求，没有明显的立群点的特征。

### Variants:

1. How does logistic regression handle binary classification?
2. What is the difference between One-vs-Rest (OvR) and Softmax regression for multi-class problems?
3. Can logistic regression handle multi-class classification natively?

### Key Comparisons:

1. Compare logistic regression with Support Vector Machines (SVM).
2. What are the advantages and disadvantages of logistic regression compared to decision trees or random forests?

------

## **2. Mathematical Questions**

### Core Equations:

1. Derive the hypothesis function $$σ(z)=\frac{1}{1 + e^{-z}}σ(z)$$ used in logistic regression.
2. Explain the equation for the decision boundary in logistic regression.
3. How is the probability of an event modeled in logistic regression?

### Loss Function:

1. What is the loss function in logistic regression? Why is the log-loss used instead of mean squared error (MSE)?



**Answer:** The loss function used in logistic regression is the **log-loss** or **binary cross-entropy**. The log-loss for a single data point is:

$$L(y,h)= - \left[ y \log(h) + (1 - y) \log(1 - h) \right]$$

Where:

- y is the true label (0 or 1),
- h is the predicted probability (the output of the Sigmoid function).

The log-loss is preferred over MSE because it ensures the probability prediction is close to the actual class labels. The log-loss penalizes large deviations in predicted probabilities more heavily than MSE.

1. Derive the log-loss (negative log-likelihood) function for logistic regression.
2. Why is the log-loss function convex, and why is that useful?

### Optimization:

1. How are the parameters www and bbb in logistic regression learned?
2. Explain how gradient descent works in logistic regression.
3. What is stochastic gradient descent (SGD), and how does it differ from batch gradient descent?

------

## **3. Regularization and Overfitting**

### Regularization Techniques:

1. Why is regularization important in logistic regression?

**Answer:** Regularization is important to prevent **overfitting**, especially when there are many features or when the features are highly correlated. By adding a penalty term to the loss function, regularization controls the size of the coefficients and helps improve the model's generalization ability.

- **L1 regularization (Lasso)** encourages sparsity, driving some coefficients to zero.
- **L2 regularization (Ridge)** penalizes large coefficients without forcing them to zero.

1. What is the difference between L1 and L2 regularization in logistic regression?
2. How does L1 regularization result in feature selection?
3. Explain the trade-off between bias and variance in logistic regression.

### Overfitting Prevention:

1. How can you prevent overfitting in logistic regression?
2. What is the impact of feature scaling on logistic regression? Why is it important?

------

## **4. Data-Related Questions**

### Data Preprocessing:

1. How does logistic regression handle missing data?
2. Why is feature scaling important in logistic regression?
3. Can logistic regression work with categorical features directly? If not, how would you preprocess them?

### Imbalanced Data:

1. What are the challenges of applying logistic regression to imbalanced datasets?
2. How can you address class imbalance in logistic regression (e.g., weighted loss, resampling)?

### Feature Engineering:

1. How does multicollinearity affect logistic regression? How can it be resolved?
2. Is feature interaction important in logistic regression? How would you incorporate it?

------

## **5. Practical and Applied Questions**

### Model Performance:

1. How do you evaluate the performance of a logistic regression model?
2. What are some common metrics for binary classification using logistic regression (e.g., accuracy, precision, recall, F1-score, ROC-AUC)?
3. What does the confusion matrix tell you about the performance of a logistic regression model?

### Model Interpretation:

1. How do you interpret the coefficients in logistic regression?
2. What does the magnitude and sign of a logistic regression coefficient indicate?
3. How can odds ratios be derived from logistic regression coefficients?

### Troubleshooting:

1. If a logistic regression model performs poorly, what steps would you take to improve it?
2. Why might a logistic regression model predict probabilities very close to 0 or 1?

------

## **6. Algorithmic Limitations**

### Linearity and Nonlinearity:

1. Can logistic regression model nonlinear decision boundaries? If not, how would you address this?
2. How does logistic regression perform with linearly separable vs. non-linearly separable data?

### Alternatives:

1. What are some alternatives to logistic regression for classification tasks?
2. Why might you choose logistic regression over more advanced algorithms like neural networks?

------

## **7. Open-Ended Questions**

1. In which situations would logistic regression fail?
2. How would you use logistic regression for a business application (e.g., churn prediction, fraud detection)?
3. What are some practical examples of using logistic regression in industry?
4. Explain a situation where logistic regression was your first choice and why.
5. Discuss a time when logistic regression failed for a project and how you handled it.

------

## **8. Advanced Topics**

### Probabilistic Interpretation:

1. Explain the concept of likelihood and how it is maximized in logistic regression.
2. What does the predicted probability mean in logistic regression?

### Regularization Fine Points:

1. What is elastic net regularization, and how does it combine L1 and L2 regularization?
2. How would you tune the regularization strength in logistic regression?

### Advanced Optimizations:

1. What is Newton’s Method? How does it differ from gradient descent for logistic regression?
2. What are some second-order optimization techniques used in logistic regression (e.g., quasi-Newton methods)?

------

## **9. Coding Questions**

### Implementation:

1. Write a Python implementation of logistic regression from scratch (including gradient descent).
2. How would you use Scikit-learn to build a logistic regression model? Demonstrate key hyperparameters.

### Debugging:

1. How would you debug a logistic regression model that converges too slowly?
2. What are some practical ways to speed up logistic regression training on large datasets?







