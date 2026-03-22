import numpy as np
try:
    from statsmodels.tsa.vector_ar.var_model import VAR
    # Generate dummy multivariate series
    T, H = 50, 8
    series = np.random.randn(T, H)
    model = VAR(series)
    res = model.fit(maxlags=3)
    print(f"VAR success. Coeffs shape: {res.coefs.shape}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")
