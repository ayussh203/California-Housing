import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def pretty_print_linear(coefs, names=None, sort=False):
    if names is None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)

def load_data():
    california = fetch_california_housing()
    X = california["data"]
    Y = california["target"]
    names = california["feature_names"]
    return X, Y, names

def scale_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def split_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    return X_train, X_test, Y_train, Y_test

def root_mean_square_error(y_pred, y_test):
    rmse_train = np.sqrt(np.dot(abs(y_pred - y_test), abs(y_pred - y_test)) / len(y_test))
    return rmse_train

def plot_real_vs_predicted(y_pred, y_test):
    plt.plot(y_pred, y_test, 'ro')
    plt.plot([0, 5], [0, 5], 'g-')
    plt.xlabel('predicted')
    plt.ylabel('real')
    plt.show()
    return plt

def generate_regression_values(model, X, y):
    params = np.append(model.intercept_, model.coef_)
    predictions = model.predict(X)
    newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3[
        "Probabilites"
    ] = [params, sd_b, ts_b, p_values]
    print(myDF3)

X, Y, names = load_data()

X = scale_data(X)
X_train, X_test, Y_train, Y_test = split_data(X, Y)

# Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, Y_train)
print("Linear model: ", pretty_print_linear(linreg.coef_, names, sort=True))
Y_lin_predict = linreg.predict(X_test)
print("Root Mean Square Error: {}".format(root_mean_square_error(Y_lin_predict, Y_test)))
plot_real_vs_predicted(Y_test, Y_lin_predict)
generate_regression_values(linreg, X_test, Y_test)

# Lasso Regression
lasso = Lasso(alpha=.3)
lasso.fit(X_train, Y_train)
print("Lasso model: ", pretty_print_linear(lasso.coef_, names, sort=True))
Y_lasso_predict = lasso.predict(X_test)
print("Root Mean Square Error: ", root_mean_square_error(Y_lasso_predict, Y_test))
plot_real_vs_predicted(Y_test, Y_lasso_predict)
generate_regression_values(lasso, X_test, Y_test)

# Ridge Regression
ridge = Ridge(fit_intercept=True, alpha=.3)
ridge.fit(X_train, Y_train)
print("Ridge model: ", pretty_print_linear(ridge.coef_, names, sort=True))
Y_ridge_predict = ridge.predict(X_test)
print("Root Mean Square Error: ", root_mean_square_error(Y_ridge_predict, Y_test))
plot_real_vs_predicted(Y_test, Y_ridge_predict)
generate_regression_values(ridge, X_test, Y_test)

# Elastic Net Regression
elnet = ElasticNet(fit_intercept=True, alpha=.3)
elnet.fit(X_train, Y_train)
print("Elastic Net model: ", pretty_print_linear(elnet.coef_, names, sort=True))
Y_elnet_predict = elnet.predict(X_test)
print("Root Mean Square Error: ", root_mean_square_error(Y_elnet_predict, Y_test))
plot_real_vs_predicted(Y_test, Y_elnet_predict)

# Stochastic Gradient Descent Regression
sgdreg = SGDRegressor(penalty='l2', alpha=0.15, max_iter=200)
sgdreg.fit(X_train, Y_train)
print("Stochastic Gradient Descent model: ", pretty_print_linear(sgdreg.coef_, names, sort=True))
Y_sgdreg_predict = sgdreg.predict(X_test)
print("Root Mean Square Error: ", root_mean_square_error(Y_sgdreg_predict, Y_test))
plot_real_vs_predicted(Y_test, Y_sgdreg_predict)