# Face Detection Flow — End-to-End

This doc explains what was implemented for **high-accuracy face detection**, **persistent bounding box until the person leaves**, and the full flow with diagram and example.

---

## 1. Summary of Changes

| Change | Where | What |
|--------|--------|------|
| **ArcFace for accuracy** | `app/utils/face_embedding.py` | `DEFAULT_EMBEDDING_MODEL = "ArcFace"` (was Facenet). Better recognition; gallery must use ArcFace (re-upload if you had Facenet). |
| **Box until person leaves** | `app/processing/vision_tasks/tasks/face_detection/scenario.py` | Track `consecutive_frames_without_face`. Box stays until 15 consecutive frames have **no face**; then overlay clears. No more 1.5s time-based drop. |
| **Knowledge base** | `app/knowledge_base/vision_rule_knowledge_base.json` | Face detection description updated: ArcFace, re-upload note, box-until-leave behaviour. |

---

## 2. High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FACE DETECTION END-TO-END FLOW                            │
└─────────────────────────────────────────────────────────────────────────────────┘

  [1] GALLERY SETUP (one-time per person)
  ─────────────────────────────────────
       User uploads photos (e.g. sachin/0.jpeg, 1.jpeg, …)
              │
              ▼
       POST /person-gallery/upload  (person_gallery_controller)
              │
              ▼
       face_embedding.py: ArcFace embeddings computed, stored in MongoDB
       (person doc: name, embedding_model="ArcFace", embedding[], status=active)

  ───────────────────────────────────────────────────────────────────────────────

  [2] AGENT / RULE CREATION
  ─────────────────────────
       User: "Alert me if sachin appears on camera 1"
              │
              ▼
       Agent creates rule: rule_id=face_detection, watch_names=["sachin"], camera_id, …

  ───────────────────────────────────────────────────────────────────────────────

  [3] LIVE STREAM PROCESSING (per frame)
  ─────────────────────────────────────
       Camera frame (e.g. 2 fps)
              │
              ▼
       pipeline.run()  →  face_detection scenario process(frame)
              │
              ├─► _load_gallery(): load persons with embedding_model=ArcFace, status=active
              │   (ids + names + embeddings into _known_encodings)
              │
              ├─► get_face_embeddings_from_frame_multi_detector()  (RetinaFace → MTCNN → OpenCV)
              │   → face_boxes_embeddings: [ (box, embedding), … ]
              │
              ├─► For each face: cosine similarity vs gallery; best match if ≥ min_similarity
              │   → recognized_faces = [ { box, name }, … ]
              │
              ├─► If recognized and name in watch_names → emit event (alert) once per cooldown
              │
              ├─► Overlay logic:
              │   • If current frame has faces → show current recognized_faces (boxes + names)
              │   • If current frame has 0 faces:
              │       - consecutive_frames_without_face += 1
              │       - If consecutive_frames_without_face < 15 → keep showing last_recognized_faces
              │       - If ≥ 15 → clear last_recognized_faces (person left) → no box
              │   • If current frame has ≥1 face → consecutive_frames_without_face = 0
              │
              ▼
       pipeline builds overlay.face_recognitions from scenario.get_overlay_data()
              │
              ▼
       WebSocket sends overlay (boxes + labels) to frontend → UI draws boxes

  ───────────────────────────────────────────────────────────────────────────────

  [4] WHEN PERSON LEAVES
  ──────────────────────
       Frames with no face: 1, 2, 3 … 14 → still show last box (person “still there”)
       Frame 15 with no face → clear overlay → box disappears (person left)
```

---

## 3. Example Timeline

- **Camera 1**, rule: *Alert if sachin appears*, cooldown 10 s, 2 fps.

| Time | Frame | Faces in frame | What happens | Overlay |
|------|--------|-----------------|--------------|---------|
| T+0  | 0  | 1 (sachin)   | Match; alert sent; last_recognized_faces = [sachin]; consecutive=0 | Box “sachin” |
| T+0.5| 1  | 0             | consecutive=1; 1 < 15 → show last | Box “sachin” |
| T+1  | 2  | 0             | consecutive=2; 2 < 15 → show last | Box “sachin” |
| …    | …  | 0             | … consecutive 3..14 … | Box “sachin” |
| T+7  | 14 | 0             | consecutive=14; 14 < 15 → show last | Box “sachin” |
| T+7.5| 15 | 0             | consecutive=15 → clear last_recognized_faces | No box (person left) |
| T+8  | 16 | 1 (sachin)   | Match again; consecutive=0; alert only if cooldown passed | Box “sachin” |

So: **box stays until the person is “gone” for 15 consecutive frames** (~7.5 s at 2 fps), then disappears.

---

## 4. File Reference

- **Embedding model (ArcFace):** `app/utils/face_embedding.py` — `DEFAULT_EMBEDDING_MODEL`, used for both gallery and live.
- **Gallery load + match + overlay state:** `app/processing/vision_tasks/tasks/face_detection/scenario.py` — `_load_gallery()`, `process()`, `get_overlay_data()`, `reset()`, `CONSECUTIVE_FRAMES_TO_CLEAR = 15`.
- **Overlay payload:** pipeline builds `face_recognitions` from face_detection `get_overlay_data()`; `streaming_controller` sends it over WebSocket.
- **Config:** `app/processing/vision_tasks/tasks/face_detection/config.py` — `min_similarity` (fallback from `tolerance`).

---

## 5. Accuracy and Re-upload

- **ArcFace** is used for higher accuracy than Facenet.
- Gallery and live comparison **must use the same model**. New uploads get `embedding_model="ArcFace"`. Existing **Facenet** gallery entries are not used (scenario loads by `embedding_model`). So **re-upload photos** for each person if you had Facenet before; then flow and diagram above apply as-is.
