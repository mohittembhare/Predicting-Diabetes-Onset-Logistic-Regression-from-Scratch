# Predicting Diabetes Onset: Logistic Regression from Scratch

##  Project Overview
This repository contains a statistical analysis of the Pima Indians Diabetes dataset, predicting the probability of diabetes onset using Logistic Regression. 

##  The Dataset
* **Source:** National Institute of Diabetes and Digestive and Kidney Diseases (via the UCI Machine Learning Repository).
* **Observations:** 768 adult females of Pima Indian heritage.
* **Dependent Variable:** `Outcome` (Binary: 1 = Tested positive for diabetes, 0 = Tested negative).
* **Independent Variables (Predictors):** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.

##  Mathematical Methodology

### 1. Data Cleaning (Pandas)
Biologically impossible `0` values (e.g., a Blood Pressure or BMI of 0) were treated as missing data. These were converted to `NaN` and imputed using the median of their respective columns to maintain distribution shape without introducing external library dependencies.

### 2. Matrix Preparation
To ensure the gradient descent algorithm converges properly, the independent variables matrix ($X$) was standardized (mean = 0, standard deviation = 1). An intercept column of ones was manually appended to the matrix to calculate $\beta_0$.

### 3. Logistic Regression Algorithm (NumPy)
The model estimates the probability of a positive diagnosis using the logistic (sigmoid) function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z$ is the linear combination of the predictors and their coefficients:

$$z = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k$$

The coefficients ($\beta$) were estimated using **Gradient Descent**, iteratively updating the values to minimize the log-loss error mathematically:

$$\beta := \beta - \alpha \frac{1}{n} X^T (\hat{y} - y)$$

*(Where $\alpha$ is the step size, $n$ is the number of observations, and $\hat{y}$ is the predicted probability).*

##  Key Findings & Interpretations

*The manual gradient descent algorithm converged successfully, yielding the following coefficients and odds ratios. Note: Because the independent variables were standardized prior to modeling, interpretations are based on a **one standard deviation increase**.*

### 1. The Strongest Predictors
* **Glucose (OR = 3.160):** Plasma glucose concentration is by far the strongest predictor of diabetes onset in this dataset. Holding all other variables constant, a 1 standard deviation increase in Glucose multiplies the odds of a positive diabetes diagnosis by **3.16** (a ~216% increase in odds).
* **BMI (OR = 1.908):** Body Mass Index is the second strongest risk factor. A 1 standard deviation increase in BMI nearly doubles the odds of testing positive (an ~91% increase).
* **Pregnancies (OR = 1.522):** The number of pregnancies is also a significant positive indicator, increasing the odds of a positive diagnosis by ~52% per standard deviation increase.

### 2. Moderate & Negligible Predictors
* **Diabetes Pedigree & Age (OR = 1.336 and 1.165):** Both family history (Pedigree) and Age show a positive correlation with diabetes onset, though their independent effects are smaller when accounting for Glucose and BMI.
* **Skin Thickness (OR = 1.032):** The odds ratio is extremely close to 1, indicating that Skin Thickness has almost no independent effect on the odds of diabetes in this model.
* **Blood Pressure & Insulin (OR = 0.893 and 0.904):** These variables show slight negative coefficients. In a multivariate model, this often suggests multicollinearity or that their predictive power is already being captured by other dominant variables like BMI and Glucose.

### 3. Baseline Model (Intercept)
* **Intercept (Log-Odds = -0.8611):** When all standardized continuous variables are exactly at their mean (a value of 0), the baseline log-odds of having diabetes is negative. This translates to an overall baseline probability of less than 50% for an "average" individual in this dataset.
