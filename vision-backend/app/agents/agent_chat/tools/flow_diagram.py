from __future__ import annotations

from typing import Any, Dict, List

from ....domain.models.agent import Agent
from .knowledge_base import get_rule


def create_diagram_node(
    name: str,
    label: str,
    shape_type: str = "process",
    x: float = 0.0,
    y: float = 0.0,
) -> Dict[str, Any]:
    if shape_type in ("start", "end"):
        symbol, size, color, border = "oval", [140, 60], "#f8f9fa", "#6c757d"
    elif shape_type == "decision":
        symbol, size, color, border = "diamond", [180, 100], "#fefefe", "#dc3545"
    else:
        symbol, size, color, border = "roundRect", [200, 70], "#ffffff", "#495057"
    return {
        "name": name,
        "symbol": symbol,
        "label": {"show": True, "formatter": label, "fontSize": 11, "color": "#212529", "position": "inside"},
        "symbolSize": size,
        "itemStyle": {"color": color, "borderColor": border, "borderWidth": 2, "borderRadius": 30 if symbol == "oval" else 8},
        "x": x, "y": y, "position": [x, y], "shape": symbol,
    }


def generate_agent_flow_diagram(agent: Agent) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    fps = getattr(agent, "fps", None) or 1
    cam_id = (getattr(agent, "camera_id", "") or "").strip()
    cam_display = f"{cam_id[:12]}..." if len(cam_id) > 12 else (cam_id or "N/A")
    model_display = (getattr(agent, "model", None) or "YOLOv8").replace(".pt", "")
    rules = getattr(agent, "rules", None) or []
    rule_ids = [str(r.get("rule_id") or r.get("type", "")) for r in rules if isinstance(r, dict) and (r.get("rule_id") or r.get("type"))]
    rule_label = "Condition Check"
    if len(rule_ids) == 1:
        try:
            rule_label = f"Match Rule:\n{get_rule(rule_ids[0]).get('rule_name', rule_ids[0])}"
        except Exception:
            rule_label = f"Match Rule:\n{rule_ids[0]}"
    elif len(rule_ids) > 1:
        rule_label = f"Check {len(rule_ids)} Rules"
    run_mode = getattr(agent, "run_mode", None) or "continuous"
    x_center, y_start, y_gap = 300.0, 50.0, 120.0
    y = y_start

    nodes.append(create_diagram_node("start", "START", "start", x_center, y))
    y += y_gap
    nodes.append(create_diagram_node("init", f"Initialize Camera\n{cam_display}", "process", x_center, y))
    y += y_gap

    if run_mode == "continuous":
        nodes.append(create_diagram_node("stop_check", "User Stop?", "decision", x_center, y))
        y += y_gap
        nodes.append(create_diagram_node("process_frame", f"AI Processing\n{model_display}", "process", x_center, y))
        y += y_gap
        nodes.append(create_diagram_node("rule_check", rule_label, "decision", x_center, y))
        y_branch = y + y_gap
        nodes.append(create_diagram_node("alert", "TRIGGER ALERT", "process", 150.0, y_branch))
        nodes.append(create_diagram_node("loop_cont", "Continue Loop", "process", 450.0, y_branch))
        nodes.append(create_diagram_node("end", "STOPPED", "end", x_center, y_branch + y_gap))
        links.extend([
            {"source": "start", "target": "init"},
            {"source": "init", "target": "stop_check"},
            {"source": "stop_check", "target": "process_frame", "label": {"show": True, "formatter": "NO"}},
            {"source": "stop_check", "target": "end", "label": {"show": True, "formatter": "YES"}, "isExit": True},
            {"source": "process_frame", "target": "rule_check"},
            {"source": "rule_check", "target": "alert", "label": {"show": True, "formatter": "MATCH"}},
            {"source": "rule_check", "target": "loop_cont", "label": {"show": True, "formatter": "SKIP"}},
            {"source": "alert", "target": "stop_check", "isLoop": True},
            {"source": "loop_cont", "target": "stop_check", "isLoop": True},
        ])
    else:
        interval = getattr(agent, "interval_minutes", 5)
        nodes.append(create_diagram_node("sleep", f"Patrol Wait\n{interval} min", "process", x_center, y))
        y += y_gap
        nodes.append(create_diagram_node("window_check", "Inside Schedule?", "decision", x_center, y))
        y += y_gap
        nodes.append(create_diagram_node("stop_check", "User Stop?", "decision", x_center, y))
        y += y_gap
        nodes.append(create_diagram_node("process_frame", f"AI Processing\n{model_display}", "process", x_center, y))
        y += y_gap
        nodes.append(create_diagram_node("rule_check", rule_label, "decision", x_center, y))
        y_branch = y + y_gap
        nodes.append(create_diagram_node("alert", "TRIGGER ALERT", "process", 150.0, y_branch))
        nodes.append(create_diagram_node("loop_cont", "Continue Patrol", "process", 450.0, y_branch))
        nodes.append(create_diagram_node("end", "STOPPED", "end", x_center, y_branch + y_gap))
        links.extend([
            {"source": "start", "target": "init"},
            {"source": "init", "target": "sleep"},
            {"source": "sleep", "target": "window_check"},
            {"source": "window_check", "target": "stop_check", "label": {"show": True, "formatter": "YES"}},
            {"source": "window_check", "target": "sleep", "label": {"show": True, "formatter": "NO"}, "isLoop": True},
            {"source": "stop_check", "target": "process_frame", "label": {"show": True, "formatter": "NO"}},
            {"source": "stop_check", "target": "end", "label": {"show": True, "formatter": "YES"}, "isExit": True},
            {"source": "process_frame", "target": "rule_check"},
            {"source": "rule_check", "target": "alert", "label": {"show": True, "formatter": "MATCH"}},
            {"source": "rule_check", "target": "loop_cont", "label": {"show": True, "formatter": "SKIP"}},
            {"source": "alert", "target": "window_check", "isLoop": True},
            {"source": "loop_cont", "target": "window_check", "isLoop": True},
        ])

    return {"nodes": nodes, "links": links, "layout": "vertical", "direction": "top-to-bottom"}
