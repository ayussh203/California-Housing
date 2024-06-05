Sure, here's a detailed README description for the provided project:

---

# Housing Price Prediction with Various Regression Models

This project demonstrates the use of various regression techniques to predict housing prices using the California Housing dataset. The project utilizes multiple linear regression models, including Linear Regression, Lasso Regression, Ridge Regression, Elastic Net Regression, and Stochastic Gradient Descent Regression, to understand their performance and compare the results.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Code Explanation](#code-explanation)
5. [Usage](#usage)
6. [Results](#results)
7. [Conclusion](#conclusion)

## Introduction

The primary goal of this project is to predict housing prices based on various features in the California Housing dataset. By implementing different regression models, we can analyze their performance and determine which model provides the most accurate predictions.

## Dataset

The dataset used in this project is the California Housing dataset, which is available in the `sklearn.datasets` module. This dataset includes features such as average number of rooms, population, average income, and more.

## Dependencies

To run the code in this project, you'll need the following Python libraries:
- numpy
- pandas
- matplotlib
- scikit-learn
- statsmodels
- scipy

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels scipy
```

## Code Explanation

### 1. Importing Libraries

The necessary libraries are imported, including `numpy`, `pandas`, `matplotlib`, and several modules from `scikit-learn`, `statsmodels`, and `scipy`.

### 2. Helper Functions

- `pretty_print_linear(coefs, names=None, sort=False)`: This function formats and prints the coefficients of the regression model in a readable way.
- `load_data()`: This function loads the California Housing dataset.
- `scale_data(X)`: This function standardizes the features by removing the mean and scaling to unit variance.
- `split_data(X, Y)`: This function splits the dataset into training and testing sets.
- `root_mean_square_error(y_pred, y_test)`: This function calculates the Root Mean Square Error (RMSE) between the predicted and actual values.
- `plot_real_vs_predicted(y_pred, y_test)`: This function plots the real vs. predicted values for visual comparison.
- `generate_regression_values(model, X, y)`: This function generates detailed statistics (coefficients, standard errors, t-values, and p-values) for the regression model.

### 3. Loading and Preparing Data

The dataset is loaded, standardized, and split into training and testing sets.

### 4. Implementing Regression Models

Five different regression models are implemented and evaluated:
- **Linear Regression**
- **Lasso Regression**
- **Ridge Regression**
- **Elastic Net Regression**
- **Stochastic Gradient Descent Regression**

For each model, the following steps are performed:
- Train the model using the training data.
- Predict the housing prices using the testing data.
- Calculate the RMSE.
- Plot the real vs. predicted values.
- Generate detailed regression statistics.

### 5. Example Code Usage

Here is an example of how to train and evaluate a Linear Regression model:

```python
# Linear Regression
from sklearn.linear_model import LinearRegression

# Create linear regression object
linreg = LinearRegression()

# Train the model using the training sets
linreg.fit(X_train, Y_train)

print("Linear model: ", pretty_print_linear(linreg.coef_, names, sort=True))

# Predict the values using the model
Y_lin_predict = linreg.predict(X_test)

# Print the root mean square error 
print("Root Mean Square Error: {}".format(root_mean_square_error(Y_lin_predict, Y_test)))
plot_real_vs_predicted(Y_test, Y_lin_predict)

generate_regression_values(linreg, X_test, Y_test)
```

## Usage

1. Clone this repository.
2. Install the required dependencies.
3. Run the script to train and evaluate the regression models.

```bash
python your_script_name.py
```

## Results

The performance of each regression model is evaluated based on the RMSE and visual comparison of real vs. predicted values. Detailed regression statistics are also provided for each model.

## Conclusion

This project demonstrates the implementation and comparison of various regression models for predicting housing prices. By evaluating the performance of different models, we can choose the most suitable model for our specific needs.

---

This README provides a comprehensive overview of the project, detailing each step and explaining the purpose and usage of the code.
