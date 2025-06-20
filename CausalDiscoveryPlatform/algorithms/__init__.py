# Modular algorithm system for NOTEARS causal discovery

from .base_algorithm import BaseAlgorithm
from .notears_linear import NOTEARSLinear
from .notears_nonlinear import NOTEARSNonlinear

# Try to import bnlearn algorithms with graceful fallback
try:
    from .bnlearn_algorithms import BnlearnHillClimbing, BnlearnPC, BnlearnGS
    bnlearn_algorithms = [BnlearnHillClimbing, BnlearnPC, BnlearnGS]
except ImportError:
    bnlearn_algorithms = []

from .algorithm_registry import (
    algorithm_registry,
    get_available_algorithms,
    get_algorithm,
    register_algorithm
)

__all__ = [
    'BaseAlgorithm',
    'NOTEARSLinear', 
    'NOTEARSNonlinear',
    'algorithm_registry',
    'get_available_algorithms',
    'get_algorithm',
    'register_algorithm'
] + [alg.__name__ for alg in bnlearn_algorithms]