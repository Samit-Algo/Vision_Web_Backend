# Scenarios Implementation Summary

This document describes the implementation of three new scenarios: `class_count`, `box_count`, and `restricted_zone`.

## Overview

All three scenarios have been implemented following the existing scenarios architecture pattern (similar to `weapon_detection`). The implementation is clean, simple, and follows the existing code structure.

## What Was Implemented

### 1. Object Tracking Module (`scenarios/tracking.py`)

A simple, lightweight tracking module that provides:

- **SimpleTracker**: IoU-based object tracker for maintaining object identities across frames
- **LineCrossingCounter**: Counts objects crossing a line (for entry/exit detection)
- **Track**: Represents a tracked object with history

**Key Features**:
- Simple IoU matching (no external dependencies)
- Configurable tracking parameters (max_age, min_hits, iou_threshold, score_threshold)
- Maintains object history for crossing detection

### 2. Class Count Scenario (`scenarios/class_count/`)

Counts objects of a specific class with two modes:

**Mode 1: Simple Per-Frame Counting (No Zone)**
- Counts all detections of the target class in each frame
- No tracking required
- Use case: "How many people are in the frame?"

**Mode 2: Line-Based Counting (Line Zone)**
- Tracks objects crossing a line (exactly 2 points)
- Counts entries, exits, or both
- Uses object tracking to maintain identity
- Use case: "Count how many people crossed the gate"

**Files**:
- `config.py`: Configuration parsing (target class, line zone, tracker config)
- `counter.py`: Simple per-frame detection counting
- `reporter.py`: Statistics reporting with history
- `scenario.py`: Main scenario orchestration with tracking for line-based counting

### 3. Box Count Scenario (`scenarios/box_count/`)

Identical to class_count but defaults to "box" class if not specified.

**Purpose**: Specialized for counting boxes, packages, containers, etc.

**Same features as class_count**:
- Simple per-frame counting
- Line-based crossing detection with tracking

**Files**:
- `config.py`: Configuration (defaults to "box" class)
- `scenario.py`: Scenario implementation (reuses class_count logic)

### 4. Restricted Zone Scenario (`scenarios/restricted_zone/`)

Monitors a restricted polygon zone for object presence and triggers alerts.

**Behavior**:
- Triggers alert when object of specified class is INSIDE the zone
- Uses bounding box center point to determine if object is in zone
- Alert cooldown prevents rapid-fire duplicate alerts
- Zone polygon is drawn on live streaming view

**Key Features**:
- Polygon-only (minimum 3 points required)
- Configurable alert cooldown (default: 10 seconds)
- Clears state when zone is empty
- Custom or auto-generated alert labels

**Files**:
- `config.py`: Configuration (target class, zone, cooldown)
- `zone_utils.py`: Polygon point-in-polygon detection
- `scenario.py`: Main scenario with cooldown logic

## Knowledge Base Updates

Updated both knowledge base files:
- `vision-backend/app/knowledge_base/vision_rule_knowledge_base.json`
- `Vision_Web_Backend/vision-backend/app/knowledge_base/vision_rule_knowledge_base.json`

### class_count Configuration

```json
{
  "rule_id": "class_count",
  "description": "Counts objects crossing a line using tracking, or simple per-frame",
  "zone_support": {
    "zone_type": "line",
    "supports_line": true,
    "supports_polygon": false,
    "line_for_crossing": true,
    "min_points": 2,
    "max_points": 2
  },
  "detectable_classes": ["person", "car", "truck", "bicycle", "dog", "cat"]
}
```

### box_count Configuration

```json
{
  "rule_id": "box_count",
  "description": "Counts boxes crossing a line using tracking, or simple per-frame",
  "zone_support": {
    "zone_type": "line",
    "supports_line": true,
    "supports_polygon": false,
    "line_for_crossing": true,
    "min_points": 2,
    "max_points": 2
  },
  "detectable_classes": ["box", "package", "container", "bag", "suitcase"],
  "model": "box_detection.pt"
}
```

### restricted_zone Configuration

```json
{
  "rule_id": "restricted_zone",
  "description": "Triggers alert when object is inside restricted polygon zone",
  "zone_support": {
    "zone_type": "polygon",
    "supports_polygon": true,
    "supports_line": false,
    "min_points": 3,
    "required": true
  },
  "detectable_classes": ["person", "car", "truck", "bicycle", "motorcycle", "dog"],
  "defaults": {
    "alert_cooldown_seconds": 10
  }
}
```

## Architecture Integration

### Registration

All scenarios are registered in `scenarios/__init__.py`:

```python
from app.processing.scenarios.class_count import scenario as class_count_scenario
from app.processing.scenarios.box_count import scenario as box_count_scenario
from app.processing.scenarios.restricted_zone import scenario as restricted_zone_scenario
```

The `@register_scenario()` decorator automatically registers each scenario type.

### How It Works

1. **Task Configuration**: The task/agent configuration includes a `scenarios` array
2. **Scenario Engine**: Creates scenario instances from configuration
3. **Per-Frame Processing**: Each frame is passed to all enabled scenarios
4. **Event Emission**: Scenarios emit `ScenarioEvent` objects when conditions are met
5. **Visualization**: Detection indices are returned for bounding box visualization

## Usage Examples

### Example 1: Count People Crossing a Line

```json
{
  "scenarios": [{
    "type": "class_count",
    "enabled": true,
    "class": "person",
    "label": "People Count",
    "zone": {
      "type": "line",
      "coordinates": [[100, 200], [500, 200]],
      "direction": "both"
    }
  }]
}
```

### Example 2: Count Boxes (Simple)

```json
{
  "scenarios": [{
    "type": "box_count",
    "enabled": true,
    "class": "box",
    "label": "Box Count"
  }]
}
```

### Example 3: Restricted Zone Alert

```json
{
  "scenarios": [{
    "type": "restricted_zone",
    "enabled": true,
    "class": "person",
    "label": "Unauthorized Access Alert",
    "zone": {
      "type": "polygon",
      "coordinates": [[100, 100], [500, 100], [500, 400], [100, 400]]
    },
    "alert_cooldown_seconds": 15
  }]
}
```

## Code Quality

✅ **Simple and Clean**: No complex dependencies, easy to understand
✅ **Well-Structured**: Follows existing patterns (weapon_detection)
✅ **No Linter Errors**: All files pass linting checks
✅ **Documented**: Clear docstrings and comments
✅ **Modular**: Separate files for config, logic, and utilities
✅ **Reusable**: Tracking module can be used by future scenarios

## Testing Recommendations

1. **Simple Counting**: Test class_count without zone (per-frame counting)
2. **Line Crossing**: Test with horizontal and vertical lines, both directions
3. **Polygon Zones**: Test with irregular polygons for both class_count and restricted_zone
4. **Tracking**: Verify objects maintain identity across frames
5. **Cooldown**: Verify restricted_zone doesn't spam alerts
6. **Edge Cases**: Test with no detections, multiple objects, objects entering/exiting

## Key Differences from Old Implementation

The old implementation used a `rule_engine` pattern, while this uses the `scenarios` architecture:

| Old (rule_engine) | New (scenarios) |
|-------------------|-----------------|
| Rule types | Scenario classes |
| Frame processor | Scenario engine |
| Rule state dict | Scenario _state |
| TrackingManager | SimpleTracker (embedded) |
| Helper functions | Clean OOP methods |

The new architecture is more modular, testable, and follows modern Python patterns.

## Summary

All three scenarios have been successfully implemented with:
- ✅ Clean, simple, understandable code
- ✅ Proper integration with scenarios architecture
- ✅ Updated knowledge base with detailed configurations
- ✅ Support for two counting modes (line-based with tracking, simple per-frame)
- ✅ Object tracking for line-based crossing detection
- ✅ Alert cooldown for restricted zone
- ✅ No linting errors
- ✅ Following existing code structure

**Important Note**: 
- **class_count** and **box_count** only support LINE zones (2 points) for crossing detection, NOT polygon zones
- For polygon-based presence detection, use **restricted_zone** scenario instead

The implementation is production-ready and can be tested immediately.
