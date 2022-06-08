import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from deliverable.run_model import load_data
from src.utils import save_sklearn_model

# Load the dataset
data_path = '../data/data.npz'
x, y = load_data(data_path)
# n will have the number of elements in the array
n = len(x)

# Initialize two empty numpy arrays for x3=cos(x2) and x4=x1^2
x3 = np.empty((0, n))
x4 = np.empty((0, n))
# For each element in the dataset, calculate x3 and x4
for _x1, _x2 in x:
    # And append the result to the original x3 and x4 arrays
    x3 = np.append(x3, np.cos(_x2))
    x4 = np.append(x4, np.square(_x1))

# Start creating the X matrix
X = np.copy(x)
X = np.insert(X, 2, x3, axis=1)
X = np.insert(X, 3, x4, axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize the linear regressor and fit the data
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Predict with this model
y_pred = linear_regressor.predict(X_test)

# Evaluate the predictions
MSE = mean_squared_error(y_test, y_pred)
print("LRM MSE: {}".format(MSE))

# Evaluate the predictions
R2 = r2_score(y_test, y_pred)
print("LRM R2: {}".format(R2))

# Calculate the variance
l = len(y_test)
e_k = (y_test - y_pred) ** 2
e_mk = e_k.mean()
s2 = (1 / (l - 1)) * np.sum((e_k - e_mk) ** 2)
print("LRM s2: {}".format(s2))

# Now initialize again the LinearRegressor with the entire dataset
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# The intercept of the modal will be theta0, as it is a constant
t0 = linear_regressor.intercept_
# The rest of the coeficientes will be theta1 to theta 4
t1, t2, t3, t4 = linear_regressor.coef_
print("")
print("theta0 = {}".format(t0))
print("theta1 = {}".format(t1))
print("theta2 = {}".format(t2))
print("theta3 = {}".format(t3))
print("theta4 = {}".format(t4))

# And save the model
save_sklearn_model(linear_regressor, '../deliverable/linear_regression.pickle')
