---
title: "Some Questions about logistic regression"
collection: machine_learning
type: "Undergraduate course"
permalink: /machine_learning/logistic_regression
#venue: "Beijing"
date: 2025-01-26
---



# LR

## **1. Basic Theory Questions**

### 1.1 Core Concepts:

1. *<u>What is logistic regression? How does it work?</u>*

**Answer:** Logistic regression is a supervised machine learning algorithm used for binary classification problems. It models the relationship between a dependent binary variable and one or more independent variables by using a logistic (Sigmoid) function.

The logistic function, $$σ(z)=\frac{1}{1 + e^{-z}}σ(z)$$, maps the linear combination of input features (weighted sum) to a probability value between 0 and 1. The output probability represents the likelihood of the instance belonging to the positive class (usually labeled as 1).

2. <u>Why is logistic regression a classification algorithm and not a regression algorithm?</u>

**Answer:** Logistic regression is called "regression" due to its use of a linear model (weighted sum of features) to predict outcomes. However, it’s primarily used for classification because it transforms the linear output through a **Sigmoid function**, which converts the result into a probability score between 0 and 1. This is interpreted as a probability of class membership, not a continuous value.

3. <u>How does logistic regression differ from linear regression?</u>

**Answer:**

- **Linear regression** predicts a continuous output based on a linear relationship between input features and output.
- **Logistic regression** is used for classification tasks, predicting the probability of a binary outcome, and applies a Sigmoid function to map the linear output to a value between 0 and 1.

Linear regression minimizes mean squared error (MSE), while logistic regression minimizes log-loss (cross-entropy loss).

4. <u>What is the role of the Sigmoid function in logistic regression?</u>

**Answer:** The Sigmoid function maps any real-valued number into the range (0, 1). This is useful for binary classification, as the output can be interpreted as the probability of the instance belonging to the positive class. The function is defined as:

$$σ(z)=\frac{1}{1 + e^{-z}}σ(z)$$

where zzz is the linear combination of the features.

5. Explain how logistic regression models probabilities.

6. What are some assumptions made by logistic regression?

**Answer:**

- **Linearity**: Assumes a linear relationship between the input variables and the log-odds of the outcome.
- **Independence**: Assumes that the observations are independent of each other.
- **No multicollinearity**: Assumes no perfect correlation between the independent variables.
- **No outliers**

### Variants:

1. How does logistic regression handle binary classification?
2. What is the difference between One-vs-Rest (OvR) and Softmax regression for multi-class problems?
3. Can logistic regression handle multi-class classification natively?

### Key Comparisons:

1. Compare logistic regression with Support Vector Machines (SVM).
2. What are the advantages and disadvantages of logistic regression compared to decision trees or random forests?

------

## **2. Mathematical Questions**

### 2.1 Core Equations:

1. <u>Derive the hypothesis function $$σ(z)=\frac{1}{1 + e^{-z}}σ(z)$$ used in logistic regression.</u>
2. Explain the equation for the decision boundary in logistic regression.
3. How is the probability of an event modeled in logistic regression?

### 2.2 Loss Function:

1. <u>What is the loss function in logistic regression? Why is the log-loss used instead of mean squared error (MSE)?</u>

**Answer:** The loss function used in logistic regression is the **log-loss** or **binary cross-entropy**. The log-loss for a single data point is:

$$L(y,h)= - \left[ y \log(h) + (1 - y) \log(1 - h) \right]$$

Where:

- y is the true label (0 or 1),
- h is the predicted probability (the output of the Sigmoid function).

The log-loss is preferred over MSE because it ensures the probability prediction is close to the actual class labels. The log-loss penalizes large deviations in predicted probabilities more heavily than MSE.

**<u>*Another reason is the MSE is difficult to get its Derivative, it is hard to solve*</u>** 

***<u>==however,  in the real application of the uplift modeling , these two loss is very similar for the final mertics==</u>***

1. Derive the log-loss (negative log-likelihood) function for logistic regression.
2. Why is the log-loss function convex, and why is that useful?

### 2.3 Optimization:

1. <u>How are the parameters w and b in logistic regression learned?</u>

**Answer:** Gradient Descent is an optimization algorithm used to minimize the log-loss function. In each iteration, the gradient of the loss function with respect to the parameters is computed, and the parameters are updated in the opposite direction of the gradient:

$$w = w - \alpha \frac{\partial L}{\partial w}, \quad b = b - \alpha \frac{\partial L}{\partial b}$$

Where $$\alpha$$ is the learning rate, controlling the step size. The process repeats until convergence, i.e., when the change in the loss function becomes negligible.

1. Explain how gradient descent works in logistic regression.
2. What is stochastic gradient descent (SGD), and how does it differ from batch gradient descent?

------

## **3. Regularization and Overfitting**

### 3.1 Regularization Techniques:

1. <u>Why is regularization important in logistic regression?</u>

**Answer:** Regularization is important to prevent **overfitting**, especially when there are many features or when the features are highly correlated. By adding a penalty term to the loss function, regularization controls the size of the coefficients and helps improve the model's generalization ability.

- **L1 regularization (Lasso)** encourages sparsity, driving some coefficients to zero.
- **L2 regularization (Ridge)** penalizes large coefficients without forcing them to zero.

2. What is the difference between L1 and L2 regularization in logistic regression?
3. How does L1 regularization result in feature selection?
4. Explain the trade-off between bias and variance in logistic regression.

### Overfitting Prevention:

1. How can you prevent overfitting in logistic regression?
2. What is the impact of feature scaling on logistic regression? Why is it important?

------

## **4. Data-Related Questions**

### 4.1 Data Preprocessing:

1. <u>How does logistic regression handle missing data?</u>

**Answer:** Logistic regression itself does not handle missing data. Before applying logistic regression, missing values should be handled using techniques such as:

- Imputation (e.g., filling missing values with the mean, median, or mode).
- Deleting rows or columns with missing data (if they represent a small portion of the dataset).

2. <u>Why is feature scaling important in logistic regression?</u>

**Answer:** Feature scaling is important because logistic regression uses the gradient descent optimization method, which can be slow or unstable if features have very different scales. Standardizing features (mean = 0, standard deviation = 1) helps ensure that the gradient descent converges more quickly and efficiently.

3. <u>Can logistic regression work with categorical features directly? If not, how would you preprocess them?</u>

**Answer:** Logistic regression requires numerical input features. Categorical features must be converted to a numerical form using methods like:

- **One-hot encoding**: Creates a new binary feature for each category.
- **Label encoding**: Converts categories into integer labels (suitable for ordinal data).

### 4.2 Imbalanced Data:

1. What are the challenges of applying logistic regression to imbalanced datasets?
2. How can you address class imbalance in logistic regression (e.g., weighted loss, resampling)?

### 4.3 Feature Engineering:

1. <u>How does multicollinearity affect logistic regression? How can it be resolved?</u>

*<u>How Does Multicollinearity Affect Logistic Regression?</u>*

**Answer**

**Multicollinearity** occurs when two or more independent variables in a logistic regression model are highly correlated with each other. This can create several issues that affect the model's performance and interpretation:

#### 1. **Unstable Coefficients**:

- When independent variables are highly correlated, the model has difficulty determining the individual effect of each variable on the dependent variable. As a result, the coefficients (www) can become unstable or fluctuate significantly with small changes in the data. This makes it hard to interpret the importance of each feature.

#### 2. **Inflated Standard Errors**:

- Multicollinearity increases the **standard errors** of the estimated coefficients. This means that the confidence intervals for the coefficients will be wider, and the statistical significance of the predictors might be reduced, even if they have a meaningful effect on the target variable. It becomes harder to detect whether a feature truly has an impact on the outcome.

#### 3. **Overfitting**:

- The model may fit the noise or random fluctuations in the data rather than the underlying signal, leading to overfitting. The high correlation between predictors can cause the model to focus on redundant information, which can hurt its generalizability to new, unseen data.

#### 4. **Interpretation Issues**:

- When variables are highly correlated, it becomes difficult to assess the individual contribution of each feature. You might end up with an inflated or misleading interpretation of the importance of features in the model.


### <u>***How Can Multicollinearity Be Resolved in Logistic Regression?***</u>

There are several strategies to address multicollinearity and stabilize the logistic regression model:

#### **1. Remove One of the Correlated Features**:

- **Step 1**: Identify highly correlated features by calculating the **correlation matrix** or using statistical measures like **Variance Inflation Factor (VIF)**.
- **Step 2**: Drop one of the variables in each highly correlated pair or group. This removes redundancy and reduces multicollinearity.

For example, if you have two features like "height" and "weight," which are often highly correlated, you might choose to remove one of them.

#### **2. Combine Correlated Features**:

- Instead of removing correlated features, you can combine them into a single feature. This can be done through techniques like:
  - **Principal Component Analysis (PCA)**: PCA is a dimensionality reduction technique that creates new, uncorrelated features (principal components) by combining the original features.
  - **Feature Engineering**: You can combine correlated features by applying domain knowledge to create a new feature (e.g., combining "height" and "weight" into a "body mass index" (BMI) feature).

#### **3. Regularization**:

- **L1 (Lasso) Regularization**: Lasso regularization encourages sparsity, which can automatically shrink some of the coefficients of correlated features to zero, effectively removing them from the model. This can help mitigate the effects of multicollinearity.
- **L2 (Ridge) Regularization**: Ridge regularization helps by shrinking the coefficients, but it doesn't set them to zero. This can help manage multicollinearity by reducing the impact of correlated features without removing them completely.
- **Elastic Net Regularization**: This is a combination of L1 and L2 regularization, which can perform both variable selection (like Lasso) and coefficient shrinkage (like Ridge).

#### **4. Increase Sample Size**:

- In some cases, multicollinearity can be alleviated by increasing the sample size. A larger dataset can help reduce the uncertainty associated with the coefficients and stabilize the model’s estimates.

#### **5. Use Domain Knowledge**:

- When feature correlation arises due to the nature of the data (e.g., features like "age" and "age group"), applying domain knowledge to merge or transform features can help. For example, you can create categorical variables or bins to reduce the effect of continuous feature correlation.

#### **6. Check for Interaction Effects**:

- Sometimes the correlation between features may be due to an **interaction effect** that isn't being captured in the model. Try adding interaction terms between the features to better capture the relationship between them without causing multicollinearity.

2. Is feature interaction important in logistic regression? How would you incorporate it?
3. <u>*What is feature discretization and feature crossing? Why does logistic regression need feature discretization?*</u>

**Answer** 

- **Feature discretization** refers to converting continuous numeric features into discrete categories. For example, in a credit scoring model, this could involve binning a continuous feature into categories and then mapping each bin to a Weight of Evidence (WoE) value, turning the feature into a discrete one.
- **Feature crossing**, also known as **feature interaction**, involves combining individual features using operations like multiplication, division, or Cartesian product to create new features. This helps capture non-linear relationships. For instance, one might create feature interactions between age and gender or longitude and latitude using one-hot encoding. This technique is generally used for discrete features.

In practice, continuous variables are rarely directly used in a logistic regression model. Instead, features are discretized before being added to the model, such as through binning and WoE transformation in credit scoring. The advantages of this approach include:

1. **Simplified Model**: Feature discretization simplifies the model, making it more stable and reducing the risk of overfitting.
2. **Robustness to Outliers**: Discretized features are more robust to outliers. In real-world scenarios, anomalous data that is difficult to explain is often not removed, but if the feature is not discretized, it could introduce significant noise into the model.
3. **Ease of Adjusting Features**: The process of adding or removing discretized features is simple, and sparse vector operations (such as dot products) are computationally efficient, facilitating faster model iteration.
4. **Non-linearity**: Logistic regression, being a generalized linear model, has limited expressive power. After discretization, each discrete variable has its own weight, introducing non-linearity into the model, which improves its ability to capture complex relationships.
5. **Feature Crossing**: Discretized features can be crossed to further introduce non-linearity, enhancing the model's ability to express more complex relationships.

In summary, **feature discretization** transforms continuous features into discrete ones, making it easier for logistic regression to model them. **Feature crossing** introduces non-linearity, improving the model’s expressive power.

------

## **5. Practical and Applied Questions**

### Model Performance:

1. How do you evaluate the performance of a logistic regression model?

**Answer:** Performance is typically evaluated using:

- **Accuracy**: The proportion of correct predictions.
- **Precision**: The proportion of positive predictions that are actually correct.
- **Recall**: The proportion of actual positives that are correctly predicted.
- **F1-score**: The harmonic mean of precision and recall.
- **AUC-ROC Curve**: The area under the receiver operating characteristic curve, which shows the tradeoff between true positive rate and false positive rate.

1. What are some common metrics for binary classification using logistic regression (e.g., accuracy, precision, recall, F1-score, ROC-AUC)?
2. What does the confusion matrix tell you about the performance of a logistic regression model?

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







