import numpy as np
import pandas as pd

from mlib.distribution_functions import gaussian_pdf
from mlib.utils import to_numpy
from scipy.stats import norm

class AnomalyDetector:
    def __init__(
            self, 
            desity_function = gaussian_pdf,
            epsilon = 0.01
            ) -> None:
        self.desity_function = desity_function
        self.epsilon = epsilon

    def __get_data(self, data) -> None:
        if isinstance(data, pd.DataFrame):
            # print("Pandas dataframe to numpy array")
            self.data = data.values
        elif isinstance(data, list) :
            # print("List to numpy array")
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError("data must be a numpy array or a pandas dataframe")

    def fit(self , data) -> None:
        self.data = to_numpy(data)
        if self.data.ndim > 2:
            raise ValueError("data must be a 1D or 2D array")

        self.mu = np.mean(self.data, axis=0)
        self.sigma = np.std(self.data, axis=0)
        
    def density_estimation(self, x: np.ndarray|list , mu: np.ndarray|float , sigma: np.ndarray|float) -> np.ndarray:
        # Product of PDF of all features
        return self.desity_function(x, mu, sigma).prod(axis=-1)

    def predict(self, x: np.ndarray|list , epsilon=None) -> np.ndarray:
        if epsilon is None:
            epsilon = self.epsilon

        x = to_numpy(x)

        if (x.ndim > 2):
            raise ValueError("x must be 1D/2D array with same features as training data")

        p = self.density_estimation(x, self.mu, self.sigma)
        return p < epsilon

    def Precision(self, x: np.ndarray|list=None , y: np.ndarray|list=None , y_pred: np.ndarray|list=None, epsilon=None) -> float:
        if epsilon is None:
            epsilon = self.epsilon

        x = to_numpy(x)
        y = to_numpy(y, dtype=bool)
        if y_pred is None:
            y_pred = self.predict(x, epsilon=epsilon)
        Tp = np.sum(y & y_pred)     # True and Predicted Positive
        Fp = np.sum(~y & y_pred)    # False but Predicted Positive
        return Tp / (Tp + Fp)

    def Recall(self, x: np.ndarray|list=None , y: np.ndarray|list=None , y_pred: np.ndarray|list=None, epsilon=None) -> float:
        if epsilon is None:
            epsilon = self.epsilon

        x = to_numpy(x)
        y = to_numpy(y, dtype=bool)
        if y_pred is None:
            y_pred = self.predict(x, epsilon=epsilon)
        Tp = np.sum(y & y_pred)     # True and Predicted Positive
        Fn = np.sum(y & ~y_pred)    # True but Predicted Negative
        return Tp / (Tp + Fn)

    def F1_score(self, x: np.ndarray|list=None , y: np.ndarray|list=None , y_pred: np.ndarray|list=None, epsilon=None) -> float:
        if epsilon is None:
            epsilon = self.epsilon

        x = to_numpy(x)
        y = to_numpy(y, dtype=bool)
        if y_pred is None:
            y_pred = self.predict(x, epsilon=epsilon)

        Tp = np.sum(y & y_pred)     # True and Predicted Positive
        Fp = np.sum(~y & y_pred)    # False but Predicted Positive
        Fn = np.sum(y & ~y_pred)    # True but Predicted Negative

        precision = Tp / (Tp + Fp)  # Accuracy of (Positive) Predictions among Predicted Positives
        recall = Tp / (Tp + Fn)     # Accurecy of (Positive) Predictions among True Positives
        return 2 * precision * recall / (precision + recall)    # F1 Score

    def Score(self, x: np.ndarray|list=None , y: np.ndarray|list=None , y_pred: np.ndarray|list=None , display=False, epsilon=None) -> float:
        if epsilon is None:
            epsilon = self.epsilon

        x = to_numpy(x)
        y = to_numpy(y , dtype=bool)
        if y_pred is None:
            y_pred = self.predict(x, epsilon=epsilon)
        Tp = np.sum(y & y_pred)     # True and Predicted Positive
        Fp = np.sum(~y & y_pred)    # False but Predicted Positive
        Fn = np.sum(y & ~y_pred)    # True but Predicted Negative

        precision = Tp / (Tp + Fp)  # Accuracy of (Positive) Predictions among Predicted Positives
        recall = Tp / (Tp + Fn)     # Accurecy of (Positive) Predictions among True Positives
        f1 = 2 * precision * recall / (precision + recall)    # F1 Score

        if display:
            print("\nScore:")
            print(f"Precision = {precision:.3f}")
            print(f"Recall    = {recall:.3f}")
            print(f"F1 Score  = {f1:.3f}")
        return precision, recall, f1

    def cross_validation(self, x: np.ndarray|list , y: np.ndarray|list , score=F1_score, learning_rate=0.01, max_iter=100, atol=1e-5, display=False) -> float:
        x = to_numpy(x)
        y = to_numpy(y, dtype=bool)
        # y = y == 1

        # epsilon = 0

        # f1 = self.F1_score(x, y, epsilon=epsilon)

        epsilon = self.epsilon
        for i in range(max_iter):
            # Compute predicted labels using current epsilon value
            y_pred = self.density_estimation(x, self.mu, self.sigma)
            
            # Compute loss function (binary cross-entropy)
            loss = np.mean(np.log(y_pred) * y + np.log(1 - y_pred) * (1 - y))
            
            # Compute gradient of loss function with respect to epsilon
            gradient = np.mean((1*y_pred - 1*y) * (1 - 1*y_pred) * y_pred)
            
            # Update epsilon value using gradient descent update rule
            epsilon -= learning_rate * gradient
            
            # Compute F1 score
            f1 = self.F1_score(y=y, y_pred=y_pred<epsilon, epsilon=epsilon)
            
            # Print F1 score and epsilon value
            if display:
                print(f"Iteration {i+1}, score: {f1:.4f}, epsilon: {epsilon:.4f}")

        self.epsilon = epsilon
        return epsilon

    def perameters(self):
        return [self.mu, self.sigma, self.epsilon]

    def save(self, path):
        model_data = np.array([self.mu, self.sigma, self.epsilon])
        np.save(path, model_data)

    def load(self, path):
        model_data = np.load(path)
        if model_data.ndim !=1 and model_data.size != 3:
            raise ValueError("Invalid model data: must be a 1D array with 3 elements [mu, sigma, epsilon]")
        
        self.mu = model_data[0]
        self.sigma = model_data[1]
        self.epsilon = model_data[2]
