import numpy as np
import pandas as pd
import scipy as sp
from mlib.anomaly_detection import AnomalyDetector
import pytest

mu = [4, 5, -1]
sigma = [1, 0.001, 1]
size = (50000, len(mu))
x = np.random.normal(mu, sigma, size)
data_fit = [
    ( x, mu, sigma),
]
@pytest.mark.parametrize("x , mu , sigma", data_fit)
def test_fit(x , mu , sigma):
    model = AnomalyDetector()
    model.fit(x)
    assert np.allclose(model.mu , mu, atol=1e-1)
    assert np.allclose(model.sigma , sigma, atol=1e-1)


if __name__ == "__main__":
    X = np.load("data/X_part1.npy")
    X_val = np.load("data/X_val_part1.npy")
    y_val = np.load("data/y_val_part1.npy")

    # print("X = \n" , X)
    # print("X_val = \n" , X_val.shape)
    print("y_val = \n" , y_val[:5])


    model = AnomalyDetector(epsilon=0.01)
    model.fit(X)

    print("Mean = ", model.mu , "\nSigma = " , model.sigma)
    # print("Compare Mean: ", np.allclose(model.mu , mu , atol=1e-2))
    # print("Compare Sigma: ", np.allclose(model.sigma , sigma , atol=1e-2))

    # print("PDF = \n", model.desity_function([7, 0.6, -2], model.mu, model.sigma))

    # x_test = [7, 5, -2]

    # print("Density Function = ", model.desity_function(X_val, model.mu, model.sigma))
    print("Density Estimation = ", model.density_estimation(X_val, model.mu, model.sigma).shape)

    y = model.predict(X_val)
    print("Predict = ",y.shape)

    print("Accuracy = ", np.mean(y == y_val))

    model.Score(X_val, y_val, display=True)
    # print(np.array([1,0,0]) & np.array([1,0,1]))

    model.cross_validation(X_val, y_val)

    model.Score(X_val, y_val, display=True)
    print("Epsilon = ", model.epsilon)
