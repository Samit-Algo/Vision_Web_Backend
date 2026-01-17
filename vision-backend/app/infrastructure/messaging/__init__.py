"""Messaging infrastructure for event processing"""

try:
    from .kafka_consumer import KafkaEventConsumer
except Exception as e:  # pragma: no cover
    # Keep imports optional so the app can run without kafka-python installed.
    print(f"Failed to import KafkaEventConsumer: {e}")
    KafkaEventConsumer = None  # type: ignore[assignment]

__all__ = ["KafkaEventConsumer"]

