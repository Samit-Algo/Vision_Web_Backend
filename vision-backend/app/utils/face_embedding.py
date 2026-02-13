"""
Face embedding using DeepFace (Facenet/ArcFace etc.).
Used for person gallery uploads and live face recognition in the pipeline.
"""
from typing import List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Model and detector used for both upload and live recognition (must match).
# ArcFace gives better accuracy than Facenet; gallery must have embedding_model="ArcFace" (re-upload if needed).
DEFAULT_EMBEDDING_MODEL = "ArcFace"
DEFAULT_DETECTOR_BACKEND = "opencv"

# For gallery uploads: try these detectors in order (better for front + side faces).
# RetinaFace and MTCNN handle profile/side angles better than OpenCV.
GALLERY_DETECTOR_BACKENDS = ["retinaface", "mtcnn", "opencv"]


def _get_deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except ImportError:
        return None


def compute_face_embedding(
    image_path: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND,
    enforce_detection: bool = True,
) -> Optional[List[float]]:
    """
    Compute face embedding for one image. Returns None if not exactly one face.
    Used when uploading reference photos to the person gallery.
    """
    DeepFace = _get_deepface()
    if DeepFace is None:
        logger.warning("DeepFace not available; install deepface.")
        return None
    try:
        objs = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
        )
        if not objs or len(objs) != 1:
            return None
        emb = objs[0].get("embedding")
        if emb is None:
            return None
        # Ensure native Python floats for BSON and consistency
        return [float(x) for x in emb]
    except Exception as e:
        logger.info("DeepFace.represent failed for %s: %s", image_path, e, exc_info=True)
        return None


def compute_face_embedding_for_gallery(
    image_path: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Optional[List[float]]:
    """
    Compute one face embedding for person-gallery uploads. Supports front-facing
    and side-facing photos by trying multiple detectors (RetinaFace, MTCNN, OpenCV).
    If multiple faces are detected, returns the embedding of the largest face.
    Returns None only if no face could be detected with any detector.
    """
    DeepFace = _get_deepface()
    if DeepFace is None:
        logger.warning("DeepFace not available; install deepface.")
        return None
    last_error: Optional[Exception] = None
    for detector_backend in GALLERY_DETECTOR_BACKENDS:
        try:
            objs = DeepFace.represent(
                img_path=image_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
            )
            if not objs:
                continue
            # Pick the face with largest area (main subject); same person in different angles is fine
            best = None
            best_area = 0
            for obj in objs:
                emb = obj.get("embedding")
                area = obj.get("facial_area") or {}
                w, h = int(area.get("w", 0)), int(area.get("h", 0))
                if emb is not None and w > 0 and h > 0:
                    a = w * h
                    if a > best_area:
                        best_area = a
                        best = emb
            if best is not None:
                return [float(x) for x in best]
        except Exception as e:
            last_error = e
            logger.debug("Gallery detector %s failed for %s: %s", detector_backend, image_path, e)
            continue
    if last_error:
        logger.info("No face detected in %s (tried %s): %s", image_path, GALLERY_DETECTOR_BACKENDS, last_error)
    return None


def get_face_embeddings_from_frame(
    frame_rgb: Any,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND,
    enforce_detection: bool = True,
) -> List[Tuple[List[float], List[float]]]:
    """
    Detect faces in a frame (RGB numpy array) and return (box, embedding) per face.
    box = [left, top, right, bottom]. Returns empty list if no faces or on error.
    """
    DeepFace = _get_deepface()
    if DeepFace is None:
        return []
    try:
        objs = DeepFace.represent(
            img_path=frame_rgb,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
        )
        result: List[Tuple[List[float], List[float]]] = []
        for obj in objs or []:
            emb = obj.get("embedding")
            area = obj.get("facial_area") or {}
            if emb is None:
                continue
            x = int(area.get("x", 0))
            y = int(area.get("y", 0))
            w = int(area.get("w", 0))
            h = int(area.get("h", 0))
            box = [float(x), float(y), float(x + w), float(y + h)]
            result.append((box, list(emb)))
        return result
    except Exception as e:
        logger.debug("DeepFace.represent (frame) failed: %s", e)
        return []


def get_face_embeddings_from_frame_multi_detector(
    frame_rgb: Any,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    enforce_detection: bool = True,
) -> List[Tuple[List[float], List[float]]]:
    """
    Try multiple detectors (retinaface, mtcnn, opencv) in order and return the first
    non-empty result. Gives more consistent face detection across frames than opencv alone.
    """
    for detector_backend in ["retinaface", "mtcnn", DEFAULT_DETECTOR_BACKEND]:
        try:
            result = get_face_embeddings_from_frame(
                frame_rgb,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
            )
            if result:
                return result
        except Exception as e:
            logger.debug("DeepFace detector %s failed for frame: %s", detector_backend, e)
            continue
    return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Range [-1, 1]."""
    import math
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def find_best_match(
    embedding: List[float],
    known_embeddings: List[List[float]],
    known_names: List[str],
    min_similarity: float,
) -> Optional[str]:
    """
    Find the best matching name for this embedding using cosine similarity.
    Returns the name if best similarity >= min_similarity, else None.
    """
    if not known_embeddings or not known_names or len(known_embeddings) != len(known_names):
        return None
    best_name: Optional[str] = None
    best_sim = min_similarity
    for emb, name in zip(known_embeddings, known_names):
        sim = cosine_similarity(embedding, emb)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name
