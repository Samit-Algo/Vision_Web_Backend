# Fire detection rule

## What it does

- Detects **fire**, **flame**, or **smoke** anywhere in the camera frame (no zone).
- Raises an **alert** when fire is detected, based on user-configured thresholds and cooldown.

## No zone

This rule does **not** use a zone. Any fire/flame/smoke detected in the full frame counts.  
If you need to limit detection to an area, use a different rule or combine with other logic.

## User-configurable options

| Option | Description | Default |
|--------|-------------|---------|
| **classes** / **class** | Classes to detect (e.g. `fire`, `flame`, `smoke`) | `["fire", "flame", "smoke"]` |
| **confidence_threshold** | Min detection confidence (0–1) | `0.5` |
| **confirm_frames** | Consecutive frames with fire before confirming | `2` |
| **alert_cooldown_seconds** | Min seconds between alerts | `10` |
| **label** | Custom alert label (optional) | — |

## Alert behavior

1. Detections of target classes above `confidence_threshold` are counted.
2. After **confirm_frames** consecutive frames with fire, the rule is “confirmed”.
3. An **alert** is emitted only when confirmed and not in **alert_cooldown**.
4. After an alert, no new alert is sent for **alert_cooldown_seconds**.

## Example rule (no zone)

```json
{
  "type": "fire_detection",
  "classes": ["fire", "flame", "smoke"],
  "confidence_threshold": 0.5,
  "confirm_frames": 2,
  "alert_cooldown_seconds": 10,
  "label": "Fire detected!"
}
```

Zone fields (`zone`, `zone_type`, `zone_coordinates`) are **ignored** for this rule.
