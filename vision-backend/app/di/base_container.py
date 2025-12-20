# Standard library imports
from typing import Any, Callable, Dict, Type, TypeVar, Union

TypeVarType = TypeVar('TypeVarType')


class BaseContainer:
    """Base dependency injection container with core functionality"""
    
    def __init__(self) -> None:
        self.instances: Dict[Union[Type, str], Any] = {}
        self.factories: Dict[Type, Callable] = {}
    
    def register_singleton(self, interface: Union[Type[TypeVarType], str], instance: TypeVarType) -> None:
        """Register a singleton instance (supports both types and string keys)"""
        self.instances[interface] = instance
    
    def register_factory(self, interface: Type[TypeVarType], factory: Callable[[], TypeVarType]) -> None:
        """Register a factory function"""
        self.factories[interface] = factory
    
    def get(self, interface: Union[Type[TypeVarType], str]) -> TypeVarType:
        """Get an instance of the requested type or string key"""
        # Check if singleton exists (supports both types and strings)
        if interface in self.instances:
            return self.instances[interface]
        
        # For type-based lookups, check factories
        if isinstance(interface, type):
            if interface in self.factories:
                instance = self.factories[interface]()
                return instance
            
            # Try to instantiate directly
            try:
                instance = interface()
                return instance
            except Exception:
                pass
        
        raise ValueError(f"No registration found for {interface}")

