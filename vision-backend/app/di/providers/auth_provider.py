from typing import TYPE_CHECKING
from ...domain.repositories.user_repository import UserRepository
from ...application.use_cases.auth.register_user import RegisterUserUseCase
from ...application.use_cases.auth.login_user import LoginUserUseCase
from ...application.use_cases.auth.get_current_user import GetCurrentUserUseCase

if TYPE_CHECKING:
    from ..base_container import BaseContainer


class AuthProvider:
    """Authentication use case provider - registers all auth-related use cases"""
    
    @staticmethod
    def register(container: "BaseContainer") -> None:
        """
        Register all authentication use cases.
        Use cases are created on-demand via factories.
        """
        # Register RegisterUserUseCase
        container.register_factory(
            RegisterUserUseCase,
            lambda: RegisterUserUseCase(
                user_repository=container.get(UserRepository)
            )
        )
        
        # Register LoginUserUseCase
        container.register_factory(
            LoginUserUseCase,
            lambda: LoginUserUseCase(
                user_repository=container.get(UserRepository)
            )
        )
        
        # Register GetCurrentUserUseCase
        container.register_factory(
            GetCurrentUserUseCase,
            lambda: GetCurrentUserUseCase(
                user_repository=container.get(UserRepository)
            )
        )

