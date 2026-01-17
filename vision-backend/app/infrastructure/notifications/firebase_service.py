"""
Firebase Cloud Messaging (FCM) Service
======================================

Provides push notification functionality for mobile apps using Firebase Admin SDK.
Initializes Firebase Admin SDK once and provides reusable functions for sending notifications.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

# Global Firebase app instance (initialized once)
_firebase_app: Optional[Any] = None


def _get_firebase_key_path() -> Path:
    """
    Get the path to the Firebase service account key JSON file.

    Returns:
        Path to firebase_key.json at the absolute given path
    """
    # Use the explicit absolute path provided by the user
    return Path("C:/Users/manoj/Desktop/Vision_backend_app/vision-backend/firebase_key.json")


def _initialize_firebase() -> Optional[Any]:
    """
    Initialize Firebase Admin SDK using the service account key.
    
    This function is called once and caches the Firebase app instance.
    
    Returns:
        Firebase app instance if successful, None otherwise
    """
    global _firebase_app
    
    if _firebase_app is not None:
        return _firebase_app
    
    try:
        import firebase_admin
        from firebase_admin import credentials, messaging
        
        # Check if Firebase is already initialized
        try:
            _firebase_app = firebase_admin.get_app()
            print("[firebase_service] ✅ Firebase Admin SDK already initialized")
            return _firebase_app
        except ValueError:
            # Not initialized yet, proceed with initialization
            pass
        
        # Get path to Firebase key file
        key_path = _get_firebase_key_path()
        
        if not key_path.exists():
            print(f"[firebase_service] ❌ Firebase key file not found: {key_path}")
            return None
        
        # Initialize Firebase Admin SDK
        cred = credentials.Certificate(str(key_path))
        _firebase_app = firebase_admin.initialize_app(cred)
        
        print(f"[firebase_service] ✅ Firebase Admin SDK initialized successfully")
        return _firebase_app
        
    except ImportError:
        print(
            "[firebase_service] ⚠️  firebase-admin not installed. "
            "Install with: pip install firebase-admin"
        )
        return None
    except Exception as e:
        print(f"[firebase_service] ❌ Failed to initialize Firebase: {e}")
        return None

def send_push_notification(
    token: str,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None,
    image_url: Optional[str] = None,
) -> bool:
    """
    Send a push notification to a mobile device via FCM.
    
    Args:
        token: FCM device token from the mobile app
        title: Notification title
        body: Notification body text
        data: Optional dictionary of custom data to include in the notification
        image_url: Optional URL to an image to display in the notification
        
    Returns:
        True if notification was sent successfully, False otherwise
    """
    try:
        from firebase_admin import messaging

        # Initialize Firebase if not already done
        app = _initialize_firebase()
        if app is None:
            print("[firebase_service] ❌ Cannot send notification: Firebase not initialized")
            return False

        # Build notification payload (only include image if provided)
        notification_kwargs = {
            "title": title,
            "body": body,
        }
        if image_url:
            notification_kwargs["image"] = image_url
        notification = messaging.Notification(**notification_kwargs)

        # Build Android-specific config (only include image if provided)
        android_notification_kwargs = {
            "title": title,
            "body": body,
            "channel_id": "vision_alerts",
            "priority": "high",
        }
        if image_url:
            android_notification_kwargs["image"] = image_url

        android_config = messaging.AndroidConfig(
            notification=messaging.AndroidNotification(**android_notification_kwargs),
            priority="high",
            direct_boot_ok=True,  # Allow notification even in direct boot mode (when device is locked/rebooting)
        )

        # Build iOS-specific config (APNS) - ensure it works when app is closed
        apns_config = messaging.APNSConfig(
            payload=messaging.APNSPayload(
                aps=messaging.Aps(
                    alert=messaging.ApsAlert(
                        title=title,
                        body=body,
                    ),
                    sound="default",
                    badge=1,
                    content_available=True,  # Wake app in background
                    mutable_content=True,  # Always allow app to process notification (needed for background)
                ),
            ),
        )

        # Note: For iOS, image is typically handled via notification service extension
        # The image_url in data will be available to the extension

        # Build message
        message = messaging.Message(
            token=token,
            notification=notification,
            data=data or {},
            android=android_config,
            apns=apns_config,
        )

        # Send notification
        response = messaging.send(message)
        print(f"[firebase_service] ✅ Notification sent successfully: {response}")
        return True

    except Exception as e:
        # Try to inspect the exception for UnregisteredError / InvalidArgumentError types
        try:
            from firebase_admin import messaging as _messaging
            if isinstance(e, getattr(_messaging, "UnregisteredError", type(None))):
                print(f"[firebase_service] ⚠️  Device token is unregistered: {token}")
                return False
            if isinstance(e, getattr(_messaging, "InvalidArgumentError", type(None))):
                print(f"[firebase_service] ❌ Invalid argument: {e}")
                return False
        except Exception:
            pass
        print(f"[firebase_service] ❌ Failed to send notification: {e}")
        return False


def send_push_notification_multicast(
    tokens: list[str],
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None,
    image_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send push notifications to multiple devices at once (more efficient).
    
    Args:
        tokens: List of FCM device tokens
        title: Notification title
        body: Notification body text
        data: Optional dictionary of custom data
        image_url: Optional URL to an image
        
    Returns:
        Dictionary with success/failure counts:
        {
            "success_count": int,
            "failure_count": int,
            "responses": list of response IDs
        }
    """
    if not tokens:
        return {"success_count": 0, "failure_count": 0, "responses": []}

    try:
        from firebase_admin import messaging

        # Initialize Firebase if not already done
        app = _initialize_firebase()
        if app is None:
            print("[firebase_service] ❌ Cannot send notifications: Firebase not initialized")
            return {"success_count": 0, "failure_count": len(tokens), "responses": []}

        # Build notification (only include image if provided)
        notification_kwargs = {
            "title": title,
            "body": body,
        }
        if image_url:
            notification_kwargs["image"] = image_url
        notification = messaging.Notification(**notification_kwargs)

        # Build Android config (only include image if provided)
        android_notification_kwargs = {
            "title": title,
            "body": body,
            "channel_id": "vision_alerts",
            "priority": "high",
        }
        if image_url:
            android_notification_kwargs["image"] = image_url

        android_config = messaging.AndroidConfig(
            notification=messaging.AndroidNotification(**android_notification_kwargs),
            priority="high",
            direct_boot_ok=True,  # Allow notification even in direct boot mode (when device is locked/rebooting)
        )

        # Build APNS config - ensure it works when app is closed
        apns_config = messaging.APNSConfig(
            payload=messaging.APNSPayload(
                aps=messaging.Aps(
                    alert=messaging.ApsAlert(title=title, body=body),
                    sound="default",
                    badge=1,
                    content_available=True,  # Wake app in background
                    mutable_content=True,  # Always allow app to process notification (needed for background)
                ),
            ),
        )

        # Build multicast message
        message = messaging.MulticastMessage(
            tokens=tokens,
            notification=notification,
            data=data or {},
            android=android_config,
            apns=apns_config,
        )

        # Send multicast notification
        response = messaging.send_multicast(message)

        print(
            f"[firebase_service] ✅ Multicast notification sent: "
            f"{response.success_count} successful, {response.failure_count} failed"
        )

        return {
            "success_count": response.success_count,
            "failure_count": response.failure_count,
            "responses": [str(r) for r in response.responses],
        }

    except Exception as e:
        print(f"[firebase_service] ❌ Failed to send multicast notification: {e}")
        return {"success_count": 0, "failure_count": len(tokens), "responses": []}
