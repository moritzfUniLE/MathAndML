"""
Algorithm registry and factory for modular causal discovery algorithms.
"""
from typing import Dict, List, Type, Any
from .base_algorithm import BaseAlgorithm
from .notears_linear import NOTEARSLinear
from .notears_nonlinear import NOTEARSNonlinear

# Try to import bnlearn algorithms with graceful fallback
try:
    from .bnlearn_algorithms import BnlearnHillClimbing, BnlearnPC, BnlearnGS
    BNLEARN_ALGORITHMS_AVAILABLE = True
except ImportError:
    BNLEARN_ALGORITHMS_AVAILABLE = False


class AlgorithmRegistry:
    """Registry for managing available causal discovery algorithms."""
    
    def __init__(self):
        self._algorithms: Dict[str, Type[BaseAlgorithm]] = {}
        self._register_default_algorithms()
    
    def _register_default_algorithms(self):
        """Register the default NOTEARS algorithms."""
        self.register_algorithm("notears_linear", NOTEARSLinear)
        self.register_algorithm("notears_nonlinear", NOTEARSNonlinear)
        
        # Register bnlearn algorithms if available
        if BNLEARN_ALGORITHMS_AVAILABLE:
            try:
                self.register_algorithm("bnlearn_hill_climbing", BnlearnHillClimbing)
                self.register_algorithm("bnlearn_pc", BnlearnPC)
                self.register_algorithm("bnlearn_gs", BnlearnGS)
                print("[INFO] bnlearn algorithms registered successfully")
            except Exception as e:
                print(f"[WARNING] Failed to register bnlearn algorithms: {e}")
        else:
            print("[INFO] bnlearn not available - skipping bnlearn algorithms")
    
    def register_algorithm(self, algorithm_id: str, algorithm_class: Type[BaseAlgorithm]):
        """
        Register a new algorithm.
        
        Args:
            algorithm_id: Unique identifier for the algorithm
            algorithm_class: Algorithm class that inherits from BaseAlgorithm
        """
        if not issubclass(algorithm_class, BaseAlgorithm):
            raise ValueError(f"Algorithm class must inherit from BaseAlgorithm")
        
        self._algorithms[algorithm_id] = algorithm_class
        print(f"[INFO] Registered algorithm: {algorithm_id}")
    
    def get_algorithm(self, algorithm_id: str) -> BaseAlgorithm:
        """
        Get an instance of the specified algorithm.
        
        Args:
            algorithm_id: Unique identifier for the algorithm
            
        Returns:
            Instance of the requested algorithm
            
        Raises:
            ValueError: If algorithm_id is not registered
        """
        if algorithm_id not in self._algorithms:
            raise ValueError(f"Algorithm '{algorithm_id}' not found. Available: {list(self._algorithms.keys())}")
        
        return self._algorithms[algorithm_id]()
    
    def list_algorithms(self) -> List[Dict[str, Any]]:
        """
        Get list of all registered algorithms with their metadata.
        
        Returns:
            List of dictionaries containing algorithm information
        """
        algorithms = []
        for algorithm_id, algorithm_class in self._algorithms.items():
            # Create temporary instance to get metadata
            instance = algorithm_class()
            algorithms.append({
                'id': algorithm_id,
                'name': instance.name,
                'description': instance.description,
                'parameters': instance.get_parameter_definitions(),
                'defaults': instance.get_default_parameters()
            })
        return algorithms
    
    def get_algorithm_ids(self) -> List[str]:
        """Get list of registered algorithm IDs."""
        return list(self._algorithms.keys())
    
    def get_algorithm_info(self, algorithm_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific algorithm.
        
        Args:
            algorithm_id: Unique identifier for the algorithm
            
        Returns:
            Dictionary containing algorithm metadata
        """
        if algorithm_id not in self._algorithms:
            raise ValueError(f"Algorithm '{algorithm_id}' not found")
        
        instance = self._algorithms[algorithm_id]()
        return {
            'id': algorithm_id,
            'name': instance.name,
            'description': instance.description,
            'parameters': instance.get_parameter_definitions(),
            'defaults': instance.get_default_parameters()
        }


# Global registry instance
algorithm_registry = AlgorithmRegistry()


def get_available_algorithms() -> List[Dict[str, Any]]:
    """Get list of all available algorithms."""
    return algorithm_registry.list_algorithms()


def get_algorithm(algorithm_id: str) -> BaseAlgorithm:
    """Get an algorithm instance by ID."""
    return algorithm_registry.get_algorithm(algorithm_id)


def register_algorithm(algorithm_id: str, algorithm_class: Type[BaseAlgorithm]):
    """Register a new algorithm."""
    algorithm_registry.register_algorithm(algorithm_id, algorithm_class)