from mlib.distribution_functions import gaussian_pdf
import numpy as np
import scipy as sp
import pytest

g_testdata = [
    (0, 0, 1, 0.39894), # Test 1
    (   # Test 2
        np.array([0, 2, 3])
        , 0 # mu
        , 1 # sigma
        , np.array([0.39894, 0.05399, 0.00443]) ),
    (   # Test 3
        np.array([
            [0, 2, 3],
            [0, 2, 3],
            [0, 2, 3]
            ])
        , 0 # mu
        , 1 # sigma
        , np.array([
            [0.39894, 0.05399, 0.00443],
            [0.39894, 0.05399, 0.00443],
            [0.39894, 0.05399, 0.00443]
            ]) 
        ),
    (   # Test 4
        np.array([
            [0, 2, 3],
            [0, 2, 3],
            [0, 2, 3]
            ])
        , np.array([0, 0, 0]) # mu
        , np.array([1, 1, 1]) # sigma
        , np.array([
            [0.39894, 0.05399, 0.00443]
            ]) 
        ),
    (   # Test 5
        np.array([
            [0, 2, 3],
            [0, 2, 3],
            [0, 2, 3]
            ])
        , np.array([0, 0, 0]) # mu
        , np.array([1, 0, 1]) # sigma
        , np.array([
            [0.39894, 0, 0.00443]
            ]) 
        ),
    (   # Test 6
        np.array([
            [0, 2, 3],
            [0, 2, 3],
            [0, 2, 3]
            ])
        , np.array([0, 2, 0]) # mu
        , np.array([1, 0, 1]) # sigma
        , np.array([
            [0.39894, 0, 0.00443]
            ]) 
        ),
    (   # Test 7
        np.array([
            [0, 2, 3]
            ])
        , np.array([0, 2, 0]) # mu
        , np.array([1, 7, 1]) # sigma
        , np.array([
            [0.39894, 0, 0.00443]
            ]) 
        ),
    
]
@pytest.mark.parametrize("x , mu , sigma, y", g_testdata)
def test_gaussian_pdf(x , mu , sigma, y):
    sigma = np.array([1, 0, 1], dtype=float) # sigma
    gpdf = gaussian_pdf(x, mu, sigma)
    sigma[sigma == 0] = np.finfo(float).eps
    y = np.nan_to_num(sp.stats.norm.pdf(x, mu, sigma))
    # sigma = to_numpy(sigma)
    # sigma[sigma == 0] = np.finfo(float).eps
    assert np.allclose(gpdf, y , atol=1e-5 , equal_nan=True)

@pytest.mark.parametrize("x , mu , sigma, y", g_testdata)
def test_gaussian_pdf_nan(x , mu , sigma, y):
    gpdf = gaussian_pdf(x, mu, sigma , zero_approx=False)
    y = sp.stats.norm.pdf(x, mu, sigma)
    # sigma = to_numpy(sigma)
    # sigma[sigma == 0] = np.finfo(float).eps
    assert np.allclose(gpdf, y , atol=1e-5, equal_nan=True)


