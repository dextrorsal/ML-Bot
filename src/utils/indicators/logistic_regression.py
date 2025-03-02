# src/utils/indicators/logistic_regression.py

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def minimax(value, data_window):
    """
    Scale 'value' into the range defined by the min and max of 'data_window'.
    If min equals max, return the original value.
    """
    lo = np.min(data_window)
    hi = np.max(data_window)
    if np.isclose(hi, lo):
        return value
    return (hi - lo) * (value - lo) / (hi - lo) + lo

def normalize_array(arr):
    """
    Normalize a 1D array to the range [0,1] using min-max scaling.
    """
    lo = np.min(arr)
    hi = np.max(arr)
    if np.isclose(hi, lo):
        return arr  # avoid division by zero
    return (arr - lo) / (hi - lo)

class SingleDimLogisticRegression:
    """
    Replicates the Pine Script's single-weight logistic regression logic.
    For each bar, using 'lookback' points of data (X and Y),
    it fits a single weight 'w' via gradient descent and returns (loss, prediction),
    where prediction is the sigmoid output on the last data point.
    """
    def __init__(self, lookback=3, learning_rate=0.0009, iterations=1000):
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit_single_bar(self, X, Y):
        """
        Fit a single-weight logistic regression on the provided arrays X and Y,
        each of length 'lookback'. Returns a tuple (loss, pred), where 'pred' is
        the final sigmoid output computed on the last element of the normalized X.
        """
        # Normalize X and Y to [0, 1]
        X_norm = normalize_array(X)
        Y_norm = normalize_array(Y)
        p = len(X_norm)
        w = 0.0  # initialize the single weight
        for _ in range(self.iterations):
            z = w * X_norm  # elementwise multiplication
            h = sigmoid(z)
            loss = 0.0
            grad = 0.0
            for i in range(p):
                h_i = np.clip(h[i], 1e-10, 1 - 1e-10)  # avoid log(0)
                loss += -(Y_norm[i] * np.log(h_i) + (1 - Y_norm[i]) * np.log(1 - h_i))
                grad += (h[i] - Y_norm[i]) * X_norm[i]
            loss /= p
            grad /= p
            w -= self.learning_rate * grad
        final_pred = sigmoid(w * X_norm[-1])
        return (loss, final_pred)
