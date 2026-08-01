import pandas as pd
import numpy as np

# 1. Load the Data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# 2. Simple Data Cleaning
# Replace 0s with NaN, then fill with the median of each column
cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_clean] = df[cols_to_clean].replace(0, np.nan)
df[cols_to_clean] = df[cols_to_clean].fillna(df[cols_to_clean].median())

# 3. Prepare the Matrices (X and y)
# Separate the independent variables (X) and the dependent variable (y)
X_data = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Standardize the X variables (subtract mean, divide by standard deviation)
# This prevents variables with large numbers (like Insulin) from dominating the math
X_standardized = (X_data - np.mean(X_data, axis=0)) / np.std(X_data, axis=0)

# Add a column of 1s to the X matrix to act as the Intercept (Beta 0)
intercept_column = np.ones((X_standardized.shape[0], 1))
X = np.hstack((intercept_column, X_standardized))

# 4. The Statistical Math: Logistic Regression from Scratch
# Define the logit (sigmoid) function mathematically
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Set up the Gradient Descent algorithm to find the coefficients
def fit_logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    observations, variables = X.shape
    coefficients = np.zeros(variables) # Start with all Beta coefficients at 0
    
    for i in range(iterations):
        # Calculate the current prediction
        z = np.dot(X, coefficients)
        predictions = sigmoid(z)
        
        # Calculate the gradient (the direction to adjust the coefficients)
        gradient = np.dot(X.T, (predictions - y)) / observations
        
        # Update the coefficients
        coefficients -= learning_rate * gradient
        
    return coefficients

print("Running logistic regression math...")
final_coefficients = fit_logistic_regression(X, y)

# 5. Display the Results
variable_names = ['Intercept'] + list(df.columns[:-1])
results_df = pd.DataFrame({
    'Variable': variable_names,
    'Coefficient (Log-Odds)': final_coefficients,
    'Odds Ratio': np.exp(final_coefficients)
})

print("\nFinal Regression Model Results:")
print(results_df.round(4))