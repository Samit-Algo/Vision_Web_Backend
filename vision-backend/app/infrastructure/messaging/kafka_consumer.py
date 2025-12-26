"""Kafka Consumer for Vision Events

Consumes events from Kafka topic "vision-events" and stores them locally.
"""

import json
import logging
import time
import queue
from typing import Optional, Dict, Any, Tuple
from threading import Thread

try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaConsumer = None
    KafkaError = Exception

from ...core.config import get_settings
from ...utils.event_storage import save_event_from_payload, save_video_chunk_from_payload

logger = logging.getLogger(__name__)


class KafkaEventConsumer:
    """
    Kafka consumer for vision events.
    
    Consumes events from the "vision-events" topic and stores them
    to local files using the existing event storage utility.
    Also broadcasts notifications to connected WebSocket clients.
    """
    
    def __init__(self, websocket_manager: Optional[Any] = None, notification_service: Optional[Any] = None, notification_queue: Optional[queue.Queue] = None):
        """
        Initialize Kafka consumer.
        
        Args:
            websocket_manager: Optional WebSocketManager instance for broadcasting notifications
            notification_service: Optional NotificationService instance for formatting notifications
            notification_queue: Optional queue.Queue for queuing notifications from Kafka thread to main event loop (thread-safe)
        """
        if not KAFKA_AVAILABLE:
            raise ImportError(
                "kafka-python is not installed. Install it with: pip install kafka-python"
            )
        
        self.settings = get_settings()
        self.consumer: Optional[KafkaConsumer] = None
        self.is_running = False
        self._consumer_thread: Optional[Thread] = None
        self.websocket_manager = websocket_manager
        self.notification_service = notification_service
        self._notification_queue = notification_queue
        
    def _create_consumer(self) -> KafkaConsumer:
        """
        Create and configure Kafka consumer.
        
        Returns:
            Configured KafkaConsumer instance
        """
        consumer = KafkaConsumer(
            self.settings.kafka_topic,
            bootstrap_servers=self.settings.kafka_bootstrap_servers.split(','),
            group_id=self.settings.kafka_consumer_group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset=self.settings.kafka_auto_offset_reset,
            enable_auto_commit=self.settings.kafka_enable_auto_commit,
            consumer_timeout_ms=1000,  # Timeout for polling (1 second)
        )
        
        logger.info(
            f"Created Kafka consumer: topic={self.settings.kafka_topic}, "
            f"bootstrap_servers={self.settings.kafka_bootstrap_servers}, "
            f"group_id={self.settings.kafka_consumer_group_id}"
        )
        
        return consumer
    
    def _process_message(self, message: Any) -> None:
        """
        Process a single Kafka message.
        
        Handles two types of messages:
        1. event_notification: Immediate notification with single frame (sent via WebSocket)
        2. event_video: Video chunk (stored only, not sent via WebSocket)
        
        Args:
            message: Kafka message object with value, topic, partition, offset
        """
        try:
            # Extract payload from message value
            payload = message.value
            print("event is received")
            
            if not payload:
                print("received empty message")
                logger.warning(f"Received empty message from {message.topic}[{message.partition}]:{message.offset}")
                return
            
            # Determine message type
            message_type = payload.get("type", "event_notification")  # Default to event_notification for backward compatibility
            
            # Extract metadata for logging
            agent_info = payload.get("agent", {})
            camera_id = agent_info.get("camera_id", "unknown")
            agent_id = agent_info.get("agent_id", "unknown")
            event_label = payload.get("event", {}).get("label", "unknown")
            
            logger.info(
                f"Received {message_type} from Kafka: topic={message.topic}, "
                f"partition={message.partition}, offset={message.offset}, "
                f"agent_id={agent_id}, camera_id={camera_id}, label={event_label}"
            )
            
            # Handle different message types
            if message_type == "event_video":
                # Video chunk: Store only, do NOT send via WebSocket
                try:
                    saved_paths = save_video_chunk_from_payload(payload)
                    session_id = saved_paths.get("session_id", "unknown")
                    chunk_number = saved_paths.get("chunk_number", "unknown")
                    
                    logger.info(
                        f"Video chunk saved successfully: session_id={session_id}, "
                        f"chunk_number={chunk_number}, paths={saved_paths}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to save video chunk: {e}",
                        exc_info=True
                    )
                # Do not send video chunks via WebSocket
                return
            
            else:
                # event_notification: Save and send via WebSocket
                try:
                    # Save event using existing storage utility
                    saved_paths = save_event_from_payload(payload)
                    
                    logger.info(
                        f"Event saved successfully: agent_id={agent_id}, "
                        f"camera_id={camera_id}, paths={saved_paths}"
                    )
                    
                    # Broadcast notification to WebSocket clients if manager and service are available
                    if self.websocket_manager and self.notification_service:
                        try:
                            # Extract owner_user_id from payload
                            owner_user_id = self.notification_service.extract_owner_user_id(payload)
                            
                            if owner_user_id:
                                # Format notification payload
                                notification = self.notification_service.format_event_notification(
                                    payload, saved_paths
                                )
                                
                                # Queue notification for WebSocket broadcast
                                # The notification queue will be processed by the main event loop
                                if self._notification_queue:
                                    try:
                                        self._notification_queue.put((owner_user_id, notification), block=False)
                                        logger.info(
                                            f"Notification queued for user {owner_user_id} for event: agent_id={agent_id}, "
                                            f"camera_id={camera_id}, label={event_label}"
                                        )
                                    except queue.Full:
                                        logger.warning(f"Notification queue full, dropping notification for user {owner_user_id}")
                                    except Exception as e:
                                        logger.error(f"Error queueing notification: {e}", exc_info=True)
                            else:
                                logger.warning(
                                    f"Could not extract owner_user_id from event payload. "
                                    f"Notification not sent. agent_id={agent_id}, camera_id={camera_id}"
                                )
                                
                        except Exception as e:
                            # Log error but don't fail event processing
                            logger.error(
                                f"Error preparing WebSocket notification for event: {e}",
                                exc_info=True
                            )
                except Exception as e:
                    logger.error(
                        f"Failed to save event notification: {e}",
                        exc_info=True
                    )
            
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON from message {message.topic}[{message.partition}]:{message.offset}: {e}"
            )
        except ValueError as e:
            logger.error(
                f"Invalid payload structure from {message.topic}[{message.partition}]:{message.offset}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Error processing message from {message.topic}[{message.partition}]:{message.offset}: {e}",
                exc_info=True
            )
    
    def _consume_loop(self) -> None:
        """
        Main consumer loop (runs in separate thread).
        
        Continuously polls Kafka for messages and processes them.
        """
        try:
            self.consumer = self._create_consumer()
            self.is_running = True
            
            logger.info(f"Kafka consumer started, listening on topic: {self.settings.kafka_topic}")
            
            # Poll for messages continuously
            while self.is_running:
                try:
                    # Poll for messages (timeout_ms is set in consumer config)
                    message_pack = self.consumer.poll(timeout_ms=1000)
                    
                    if message_pack:
                        # Process each message from each partition
                        for topic_partition, messages in message_pack.items():
                            for message in messages:
                                self._process_message(message)
                                
                except KafkaError as e:
                    logger.error(f"Kafka error while consuming: {e}", exc_info=True)
                    # Wait a bit before retrying
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Unexpected error in consume loop: {e}", exc_info=True)
                    time.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal error in Kafka consumer: {e}", exc_info=True)
            self.is_running = False
        finally:
            self._cleanup()
    
    def start(self) -> None:
        """
        Start the Kafka consumer in a background thread.
        """
        if self.is_running:
            logger.warning("Kafka consumer is already running")
            return
        
        if self._consumer_thread and self._consumer_thread.is_alive():
            logger.warning("Kafka consumer thread is already alive")
            return
        
        logger.info("Starting Kafka consumer...")
        self._consumer_thread = Thread(
            target=self._consume_loop,
            daemon=True,  # Thread dies when main program exits
            name="KafkaEventConsumer"
        )
        self._consumer_thread.start()
        
        # Give it a moment to initialize
        time.sleep(0.5)
        
        if self.is_running:
            logger.info("Kafka consumer started successfully")
        else:
            logger.error("Failed to start Kafka consumer")
    
    def stop(self) -> None:
        """
        Stop the Kafka consumer gracefully.
        """
        if not self.is_running:
            return
        
        logger.info("Stopping Kafka consumer...")
        self.is_running = False
        
        if self._consumer_thread:
            self._consumer_thread.join(timeout=5.0)
            if self._consumer_thread.is_alive():
                logger.warning("Consumer thread did not stop within timeout")
        
        self._cleanup()
        logger.info("Kafka consumer stopped")
    
    def _cleanup(self) -> None:
        """Clean up consumer resources"""
        if self.consumer:
            try:
                self.consumer.close()
                logger.info("Kafka consumer closed")
            except Exception as e:
                logger.error(f"Error closing Kafka consumer: {e}")
            finally:
                self.consumer = None

