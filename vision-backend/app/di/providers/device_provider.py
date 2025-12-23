from typing import TYPE_CHECKING
from ...domain.repositories.device_repository import DeviceRepository
from ...application.use_cases.device.create_device import CreateDeviceUseCase
from ...application.use_cases.device.list_devices import ListDevicesUseCase
from ...application.use_cases.device.get_device import GetDeviceUseCase
from ...infrastructure.external.jetson_client import JetsonClient

if TYPE_CHECKING:
    from ..base_container import BaseContainer


class DeviceProvider:
    """Device use case provider - registers all device-related use cases"""
    
    @staticmethod
    def register(container: "BaseContainer") -> None:
        """
        Register all device use cases.
        Use cases are created on-demand via factories.
        """
        # Register JetsonClient as singleton if not already registered
        try:
            container.get(JetsonClient)
        except ValueError:
            # Not registered yet, register it now
            jetson_client = JetsonClient()
            container.register_singleton(JetsonClient, jetson_client)
        
        # Register CreateDeviceUseCase
        container.register_factory(
            CreateDeviceUseCase,
            lambda: CreateDeviceUseCase(
                device_repository=container.get(DeviceRepository),
                jetson_client=container.get(JetsonClient),
            )
        )
        
        # Register ListDevicesUseCase
        container.register_factory(
            ListDevicesUseCase,
            lambda: ListDevicesUseCase(
                device_repository=container.get(DeviceRepository),
            )
        )
        
        # Register GetDeviceUseCase
        container.register_factory(
            GetDeviceUseCase,
            lambda: GetDeviceUseCase(
                device_repository=container.get(DeviceRepository),
            )
        )

