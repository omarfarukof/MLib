import numpy as np
from mlib.utils import to_numpy
import scipy as sp


def gaussian_pdf(x , mu: np.ndarray|float , sigma: np.ndarray|float , zero_approx=True):
    if not zero_approx:
        return sp.stats.norm.pdf(x, mu, sigma)

    sigma = to_numpy(sigma)
    sigma[sigma == 0] = np.finfo(float).eps
    return sp.stats.norm.pdf(x, mu, sigma)
    

#     z = (x - mu) / sigma
#     a = (np.sqrt(2 * np.pi) * sigma)
#     ex = np.exp(-z**2 / 2)
#     return 1/a * ex



# def gaussian_pdf(x , mu: np.ndarray|float , sigma: np.ndarray|float):
#     x = to_numpy(x)
#     mu = to_numpy(mu)
#     sigma = to_numpy(sigma)
    
#     if (x.ndim == 2) and (mu.size == 1):
#         mu = mu * np.ones(x.shape[1])
#     if (x.ndim == 2) and (sigma.size == 1):
#         sigma = sigma * np.ones(x.shape[1])

#     if (sigma != 0).any() and x.ndim == 2:
#         pdf = np.zeros(x.shape , dtype=float)
#         non_zero_idx = (sigma != 0)
#         pdf[:,non_zero_idx] = 1 / (np.sqrt(2 * np.pi) * sigma[non_zero_idx]) * np.exp(-((x[:,non_zero_idx] - mu[non_zero_idx]) / sigma[non_zero_idx])**2 / 2)    
#     elif (sigma != 0).any() and x.ndim == 1:
#         pdf = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) / sigma)**2 / 2)  

#     return pdf


