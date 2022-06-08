import joblib
import numpy as np


def load_data(filename):
    """
    Loads the data from a saved .npz file.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param filename: string, path to the .npz file storing the data.
    :return: two numpy arrays:
        - x, a Numpy array of shape (n_samples, n_features) with the inputs;
        - y, a Numpy array of shape (n_samples, ) with the targets.
    """
    data = np.load(filename)
    x = data['x']
    y = data['y']

    return x, y


def evaluate_predictions(y_true, y_pred):
    """
    Evaluates the mean squared error between the values in y_true and the values
    in y_pred.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param y_true: Numpy array, the true target values from the test set;
    :param y_pred: Numpy array, the values predicted by your model.
    :return: float, the the mean squared error between the two arrays.
    """
    assert y_true.shape == y_pred.shape
    return ((y_true - y_pred) ** 2).mean()


def load_model(filename):
    """
    Loads a Scikit-learn model saved with joblib.dump.
    This is just an example, you can write your own function to load the model.
    Some examples can be found in src/utils.py.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = joblib.load(filename)

    return model


if __name__ == '__main__':
    # Load the data
    # This will be replaced with the test data when grading the assignment
    data_path = '../data/data.npz'
    x, y = load_data(data_path)

    ############################################################################
    # EDITABLE SECTION OF THE SCRIPT: if you need to edit the script, do it here
    ############################################################################

    # >>> LINEAR REGRESSION

    linear_regresion_path = './linear_regression.pickle'
    linear_regresion = load_model(linear_regresion_path)

    n = len(x)
    x3 = np.empty((0, n))
    x4 = np.empty((0, n))
    for _x1, _x2 in x:
        x3 = np.append(x3, np.cos(_x2))
        x4 = np.append(x4, np.square(_x1))
    X = np.copy(x)
    X = np.insert(X, 2, x3, axis=1)
    X = np.insert(X, 3, x4, axis=1)

    y_pred = linear_regresion.predict(X)
    mse = evaluate_predictions(y_pred, y)
    print('MSE for Linear Regression: {}'.format(mse))

    # <<< LINEAR REGRESSION

    # >>> SUPPORT VECTOR REGRESSOR

    nonlinear_regresion_path = './nonlinear_regression.pickle'
    nonlinear_regresion = load_model(nonlinear_regresion_path)
    y_pred = nonlinear_regresion.predict(x)
    mse = evaluate_predictions(y_pred, y)
    print('MSE for Non-Linear Regression: {}'.format(mse))

    # <<< SUPPORT VECTOR REGRESSOR

    # Load the trained model
    baseline_model_path = './baseline_model.pickle'
    baseline_model = load_model(baseline_model_path)

    # Predict on the given samples
    y_pred = baseline_model.predict(x)

    ############################################################################
    # STOP EDITABLE SECTION: do not modify anything below this point.
    ############################################################################

    # Evaluate the prediction using MSE
    mse = evaluate_predictions(y_pred, y)
    print('MSE: {}'.format(mse))
