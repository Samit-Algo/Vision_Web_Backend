# Report System Documentation

## Overview

This report system automatically stores counting events (entry/exit) to MongoDB for both **class_count** and **box_count** scenarios.

## How It Works

### 1. When Agent Starts
- A **summary document** is created in MongoDB (`counting_reports` collection)
- This document tracks the entire counting session
- Contains: agent details, start time, initial counts (all zeros)

### 2. When Entry/Exit Happens
- Every time an object enters or exits the counting line, an **event document** is saved
- Each event contains:
  - Track ID (unique object identifier)
  - Event type ("entry" or "exit")
  - Timestamp
  - Current counts (entry count, exit count, standby count)

### 3. When Agent Stops
- The summary document is updated with:
  - End time
  - Final entry count
  - Final exit count
  - Final standby count
  - Status changed to "completed"

## MongoDB Collection: `counting_reports`

### Event Documents (Individual Entry/Exit Events)
```json
{
  "agent_id": "agent_123",
  "agent_name": "Main Entrance Counter",
  "camera_id": "camera_001",
  "report_type": "class_count",  // or "box_count"
  "target_class": "person",
  "track_id": 5,
  "event_type": "entry",  // or "exit"
  "timestamp": "2026-01-25T10:15:30Z",
  "entry_count": 10,
  "exit_count": 8,
  "standby_count": 2,  // Objects currently in area
  "created_at": "2026-01-25T10:15:30Z"
}
```

### Summary Documents (Session Overview)
```json
{
  "agent_id": "agent_123",
  "agent_name": "Main Entrance Counter",
  "camera_id": "camera_001",
  "report_type": "class_count",
  "target_class": "person",
  "session_type": "summary",
  "start_time": "2026-01-25T09:00:00Z",
  "end_time": "2026-01-25T17:00:00Z",
  "total_entry_count": 45,
  "total_exit_count": 38,
  "final_standby_count": 7,
  "status": "completed",  // or "active"
  "created_at": "2026-01-25T09:00:00Z"
}
```

## Data Stored

### Agent Details
- `agent_id`: Unique agent identifier
- `agent_name`: Agent name
- `camera_id`: Camera ID (if available)

### Counts
- `entry_count`: Total objects that entered
- `exit_count`: Total objects that exited
- `standby_count`: Objects currently in the area (not yet exited)

### Track Information
- `track_id`: Unique ID for each tracked object
- `event_type`: "entry" or "exit"
- `timestamp`: When the event happened

## Querying Reports

### Get All Events for an Agent
```javascript
db.counting_reports.find({
  "agent_id": "agent_123",
  "report_type": "class_count",
  "session_type": { "$exists": false }  // Only event documents
})
```

### Get Session Summary
```javascript
db.counting_reports.find({
  "agent_id": "agent_123",
  "report_type": "class_count",
  "session_type": "summary"
})
```

### Get All Entry Events
```javascript
db.counting_reports.find({
  "agent_id": "agent_123",
  "event_type": "entry"
})
```

## Files

- `report_storage.py`: Contains all MongoDB storage functions
  - `save_counting_event()`: Saves individual entry/exit events
  - `initialize_report_session()`: Creates session summary when agent starts
  - `finalize_report_session()`: Updates session summary when agent stops

- `scenario.py`: Modified to call report storage functions
  - `_initialize_report_session()`: Called in `__init__` (agent starts)
  - `_save_counting_event_to_db()`: Called when entry/exit happens
  - `_finalize_report_session()`: Called in `reset()` (agent stops)

## Notes

- Reports are stored **automatically** - no manual action needed
- Each entry/exit event is saved **immediately** to MongoDB
- The system is **non-blocking** - if MongoDB fails, it won't crash the pipeline
- Both `class_count` and `box_count` use the same report storage system
- Code is **reused** - box_count imports from class_count folder
