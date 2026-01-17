"""
Kafka Event Consumer (FastAPI side)
==================================

Consumes vision event messages from Kafka and forwards them to connected
frontend clients via WebSocket.

Design goals for this project:
- Minimal and simple (no retry/reconnect loops)
- Runs inside the FastAPI process (so it can access WebSocketManager)
- Safe to run alongside the asyncio event loop (consumer runs in a thread)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, Optional

from kafka import KafkaConsumer  # type: ignore
from kafka.errors import KafkaError, NoBrokersAvailable  # type: ignore

from ...core.config import get_settings
from ...infrastructure.notifications.notification_service import NotificationService
from ...infrastructure.notifications.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class KafkaEventConsumer:
    """
    Simple Kafka consumer that forwards messages to WebSocket clients.

    Notes:
    - This consumer is meant to run in the FastAPI process.
    - It uses a background thread because kafka-python is blocking.
    """

    def __init__(
        self,
        *,
        websocket_manager: WebSocketManager,
        notification_service: NotificationService,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._ws = websocket_manager
        self._notification_service = notification_service
        self._loop = loop

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consumer: Optional[KafkaConsumer] = None

    def start(self) -> None:
        """Start the background consumer thread."""
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._run,
            name="KafkaEventConsumer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the consumer thread (best-effort) and close Kafka consumer."""
        self._stop_event.set()
        try:
            if self._consumer is not None:
                self._consumer.close()
        except Exception:
            pass

    def _run(self) -> None:
        settings = get_settings()

        try:
            self._consumer = KafkaConsumer(
                settings.kafka_topic,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                group_id=settings.kafka_consumer_group_id,
                auto_offset_reset=settings.kafka_auto_offset_reset,
                enable_auto_commit=bool(settings.kafka_enable_auto_commit),
                value_deserializer=lambda v: json.loads(v.decode("utf-8")) if v else None,
            )
        except NoBrokersAvailable as e:
            # IMPORTANT: do not crash FastAPI if Kafka is not running.
            print(f"[kafka_consumer] ⚠️ Kafka not available, consumer not started: {e}")
            return
        except Exception as e:
            print(f"[kafka_consumer] ❌ Failed to create Kafka consumer: {e}")
            return

        print(
            f"[kafka_consumer] ✅ started | topic={settings.kafka_topic} | bootstrap={settings.kafka_bootstrap_servers}"
        )

        # Poll loop (no retry logic; any fatal error exits the thread)
        try:
            while not self._stop_event.is_set():
                records = self._consumer.poll(timeout_ms=1000)
                if not records:
                    continue

                for _tp, batch in records.items():
                    for record in batch:
                        payload = record.value
                        if not isinstance(payload, dict):
                            continue
                        # Print receipt at consumer side (debug)
                        try:
                            ev = payload.get("event") or {}
                            ag = payload.get("agent") or {}
                            cam = payload.get("camera") or {}
                            meta = payload.get("metadata") or {}
                            print(
                                "[kafka_consumer] 📥 received",
                                {
                                    "label": ev.get("label"),
                                    "agent_id": ag.get("agent_id"),
                                    "camera_id": ag.get("camera_id"),
                                    "owner_user_id": cam.get("owner_user_id"),
                                    "event_id": meta.get("event_id"),
                                },
                            )
                        except Exception:
                            print("[kafka_consumer] 📥 received (payload parse failed)")
                        self._handle_payload(payload)
        except KafkaError as e:
            print(f"[kafka_consumer] ❌ Kafka consumer error, stopping: {e}")
        except Exception as e:
            print(f"[kafka_consumer] ❌ Unexpected error in Kafka consumer, stopping: {e}")
        finally:
            try:
                self._consumer.close()
            except Exception:
                pass

    def _handle_payload(self, payload: Dict[str, Any]) -> None:
        # Determine recipient user (required for send_to_user)
        owner_user_id = self._notification_service.extract_owner_user_id(payload)
        if not owner_user_id:
            print("[kafka_consumer] ⚠️ missing owner_user_id → skip websocket send")
            return

        # Keep event_id if provided (frontend expects payload.event_id)
        event_id = None
        try:
            meta = payload.get("metadata") or {}
            if isinstance(meta, dict):
                event_id = meta.get("event_id")
        except Exception:
            event_id = None

        notification = self._notification_service.format_event_notification(
            event_payload=payload,
            saved_paths=None,
            event_id=str(event_id) if event_id else None,
            include_frame_base64=False,
        )

        # Schedule async send on the FastAPI event loop.
        print(f"[kafka_consumer] 📤 scheduling websocket send | user={owner_user_id} | event_id={event_id}")
        future = asyncio.run_coroutine_threadsafe(
            self._ws.send_to_user(str(owner_user_id), notification),
            self._loop,
        )
        # We intentionally don't block (no retries / no wait).
        try:
            def _log_send_result(f: "asyncio.Future[int]") -> None:
                try:
                    exc = f.exception()
                    if exc:
                        print(f"[kafka_consumer] ❌ websocket send failed | user={owner_user_id} | err={exc}")
                        return
                    try:
                        sent_count = f.result()
                        print(f"[kafka_consumer] ✅ websocket sent | user={owner_user_id} | connections={sent_count}")
                    except Exception:
                        # ignore result parsing issues
                        pass
                except Exception:
                    return

            future.add_done_callback(_log_send_result)
        except Exception:
            pass

