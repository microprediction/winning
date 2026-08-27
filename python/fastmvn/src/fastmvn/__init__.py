"""fastmvn: fast MVN rectangle probabilities for factor-structured
covariance, and a transparent scipy accelerator.

    from fastmvn import mvn_cdf_fast          # explicit
    import fastmvn; fastmvn.patch_scipy()     # transparent (sklearnex-style)
"""
from .core import (factorize_covariance, mvn_cdf_fast,
                   mvn_cdf_fast_info)
from .patch import patch_scipy, unpatch_scipy

__all__ = ["mvn_cdf_fast", "mvn_cdf_fast_info", "factorize_covariance",
           "patch_scipy", "unpatch_scipy"]
