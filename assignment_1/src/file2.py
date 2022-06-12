import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor

from deliverable.run_model import load_data
from src.utils import save_sklearn_model

# Load the dataset
data_path = '../data/data.npz'
x, y = load_data(data_path)
# n will have the number of elements in the array
n = len(x)

# Create the X matrix
X = np.copy(x)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize the non-linear regressor and fit the data
nonlinear_regressor = SVR(kernel="rbf", gamma="auto", C=100)
nonlinear_regressor.fit(X_train, y_train)

# Predict with this model
y_pred = nonlinear_regressor.predict(X_test)

# Evaluate the predictions
MSE = mean_squared_error(y_test, y_pred)
print("NLRM MSE: {}".format(MSE))

# Evaluate the predictions
R2 = r2_score(y_test, y_pred)
print("NLRM R2: {}".format(R2))

# Calculate the variance
l = len(y_test)
e_k = (y_test - y_pred) ** 2
e_mk = e_k.mean()
s2 = (1 / (l - 1)) * np.sum((e_k - e_mk) ** 2)
print("NLRM s2: {}".format(s2))

# Now initialize again the SVR with the entire dataset
nonlinear_regressor = SVR(kernel="rbf", gamma="auto", C=100)
nonlinear_regressor.fit(X, y)

# And save the model
save_sklearn_model(nonlinear_regressor, '../deliverable/nonlinear_regression.pickle')
