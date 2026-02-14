"""
Shared constants for media uploads (video and image).

Used by API controllers (video upload, static video analysis, person gallery)
and any future code that needs the same rules. Single place for easier updates.
"""

# -----------------------------------------------------------------------------
# Video uploads (agent chat + static video analysis)
# -----------------------------------------------------------------------------
ALLOWED_VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".avi", ".mov", ".mkv"})

# Subfolder names under the configured static_video_upload_dir (see core.config)
STATIC_VIDEO_UPLOAD_SUBDIR = "static"
AGENT_VIDEO_UPLOAD_SUBDIR = "agent"

# -----------------------------------------------------------------------------
# Person gallery (reference photos for face recognition)
# -----------------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
ALLOWED_IMAGE_MIME = frozenset({"image/jpeg", "image/png", "image/webp"})
MIN_PHOTOS_PER_PERSON = 4
