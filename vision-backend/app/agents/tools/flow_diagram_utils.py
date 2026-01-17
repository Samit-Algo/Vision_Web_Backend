"""
Utility functions for generating agent processing flow diagrams.

We intentionally keep the diagram simple and readable inside the chatbot:
- Top → Bottom pipeline
- Minimal branching (rules summarized as a single step)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..session_state.agent_state import AgentState
from ...domain.models.agent import Agent
from .kb_utils import get_rule
import re


def generate_agent_flow_diagram(agent: Agent) -> Dict[str, Any]:
    """
    Generate a simple top→bottom flow diagram for a saved agent.

    Returns:
        {"nodes": [...], "links": [...]}
    """
    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []

    def create_node(
        name: str,
        label: str,
        shape: str = "rect",
        color: str = "#2A7BE4",
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> Dict[str, Any]:
        symbol = "roundRect" if shape == "rect" else "diamond"
        return {
            "name": name,
            "symbol": symbol,
            "label": {
                "show": True,
                "formatter": label,
                "fontSize": 11,
                "color": "#fff",
                "position": "inside",
            },
            "symbolSize": [170, 60] if symbol == "roundRect" else [130, 90],
            "itemStyle": {
                "color": color,
                "borderColor": "#1A5BBE",
                "borderWidth": 2,
                "borderRadius": 10 if symbol == "roundRect" else 0,
            },
            **({} if x is None or y is None else {"x": x, "y": y}),
        }

    # Vertical layout coordinates (frontend will fit/scale these to the container).
    x_center = 250.0
    y_step = 110.0
    y = 0.0

    fps = getattr(agent, "fps", None) or 1
    cam_id = (getattr(agent, "camera_id", "") or "")[:12]
    model_display = (
        getattr(agent, "model", None).replace(".pt", "")
        if getattr(agent, "model", None)
        else "YOLOv8"
    )

    # Rules summary
    rules = getattr(agent, "rules", None) or []
    rule_ids: List[str] = []
    for rc in rules:
        if isinstance(rc, dict) and rc.get("rule_id"):
            rule_ids.append(str(rc["rule_id"]))
    rule_label = "Rules\n(none)"
    if len(rule_ids) == 1:
        rid = rule_ids[0]
        try:
            rule_def = get_rule(rid)
            rn = rule_def.get("rule_name", rid)
            rule_label = f"Rules\n{rn}"
        except Exception:
            rule_label = f"Rules\n{rid}"
    elif len(rule_ids) > 1:
        rule_label = f"Rules\n{len(rule_ids)} rule(s)"

    requires_zone = bool(
        getattr(agent, "requires_zone", False) or (getattr(agent, "zone", None) is not None)
    )

    # Nodes (top → bottom)
    nodes.append(create_node("camera_stream", f"Camera\n{cam_id}...", "rect", "#DC3545", x=x_center, y=y)); y += y_step
    nodes.append(create_node("frame_extraction", f"Frame Extraction\n{fps} FPS", "rect", "#2A7BE4", x=x_center, y=y)); y += y_step
    nodes.append(create_node("ai_model", f"AI Model\n{model_display}", "rect", "#2A7BE4", x=x_center, y=y)); y += y_step
    nodes.append(create_node("object_detection", "Object\nDetection", "rect", "#2A7BE4", x=x_center, y=y)); y += y_step
    nodes.append(create_node("rule_engine", "Rule\nEngine", "diamond", "#6E7891", x=x_center, y=y)); y += y_step
    nodes.append(create_node("rules", rule_label, "diamond", "#6E7891", x=x_center, y=y)); y += y_step
    if requires_zone:
        nodes.append(create_node("zone_check", "Zone\nCheck", "diamond", "#6E7891", x=x_center, y=y)); y += y_step
    run_mode = getattr(agent, "run_mode", None)
    interval_minutes = getattr(agent, "interval_minutes", None)
    run_mode_label = "Continuous" if run_mode == "continuous" else f"Patrol\n{interval_minutes}min"
    nodes.append(create_node("run_mode", run_mode_label, "rect", "#DC3545", x=x_center, y=y)); y += y_step
    nodes.append(create_node("event_generation", "Event\nGeneration", "rect", "#28A745", x=x_center, y=y)); y += y_step
    nodes.append(create_node("kafka", "Kafka\nTopic", "rect", "#28A745", x=x_center, y=y)); y += y_step
    nodes.append(create_node("database", "MongoDB\nDatabase", "rect", "#28A745", x=x_center, y=y)); y += y_step
    nodes.append(create_node("notification", "WebSocket\nNotification", "rect", "#28A745", x=x_center, y=y)); y += y_step
    nodes.append(create_node("frontend", "Frontend\nUpdates", "rect", "#DC3545", x=x_center, y=y))

    # Links
    links.append({"source": "camera_stream", "target": "frame_extraction"})
    links.append({"source": "frame_extraction", "target": "ai_model"})
    links.append({"source": "ai_model", "target": "object_detection"})
    links.append({"source": "object_detection", "target": "rule_engine"})
    links.append({"source": "rule_engine", "target": "rules"})
    if requires_zone:
        links.append({"source": "rules", "target": "zone_check", "label": {"show": True, "formatter": "MATCH"}})
        links.append({"source": "zone_check", "target": "run_mode", "label": {"show": True, "formatter": "OK"}})
    else:
        links.append({"source": "rules", "target": "run_mode", "label": {"show": True, "formatter": "MATCH"}})
    links.append({"source": "run_mode", "target": "event_generation"})
    links.append({"source": "event_generation", "target": "kafka"})
    links.append({"source": "kafka", "target": "database"})
    links.append({"source": "database", "target": "notification"})
    links.append({"source": "notification", "target": "frontend"})

    return {"nodes": nodes, "links": links}


def generate_agent_flow_mermaid(agent: Agent) -> str:
    """
    Generate a Mermaid flow diagram (graph LR) for a saved agent.

    Output is text-only Mermaid syntax suitable for fenced blocks:
    ```mermaid
    graph LR
      ...
    ```

    This maps 1:1 to generate_agent_flow_diagram(...) output (nodes + links),
    but rendered in Mermaid instead of ECharts.
    """
    diagram = generate_agent_flow_diagram(agent)
    nodes = diagram.get("nodes") or []
    links = diagram.get("links") or []

    def _safe_id(name: str) -> str:
        # Mermaid IDs should be simple. Keep underscores; replace others.
        s = re.sub(r"[^a-zA-Z0-9_]", "_", str(name or "node"))
        if not s:
            s = "node"
        if s[0].isdigit():
            s = f"n_{s}"
        return s

    def _label_text(node: dict) -> str:
        # Our diagram nodes store label.formatter; it contains \n for multi-line.
        label = ""
        try:
            label = (node.get("label") or {}).get("formatter") or ""
        except Exception:
            label = ""
        label = str(label)
        # Shorten common labels to keep nodes compact in the chat panel.
        # (Mermaid node size is largely driven by label length.)
        replacements = {
            "Frame Extraction": "Frames",
            "AI Model": "Model",
            "Object<br/>Detection": "Detect",
            "Object\nDetection": "Detect",
            "Rule<br/>Engine": "Rules",
            "Rule\nEngine": "Rules",
            "Event<br/>Generation": "Events",
            "Event\nGeneration": "Events",
            "Kafka<br/>Topic": "Kafka",
            "Kafka\nTopic": "Kafka",
            "MongoDB<br/>Database": "MongoDB",
            "MongoDB\nDatabase": "MongoDB",
            "WebSocket<br/>Notification": "Notify",
            "WebSocket\nNotification": "Notify",
            "Frontend<br/>Updates": "UI",
            "Frontend\nUpdates": "UI",
        }
        for k, v in replacements.items():
            label = label.replace(k, v)
        # Mermaid supports <br/> for multi-line labels.
        label = label.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")
        # Escape double quotes inside labels.
        label = label.replace('"', '\\"')
        return label

    # Build node definitions so labels/shapes are preserved.
    node_lines: List[str] = []
    seen: set[str] = set()
    for n in nodes:
        raw_name = str(n.get("name") or "")
        nid = _safe_id(raw_name)
        if nid in seen:
            continue
        seen.add(nid)

        label = _label_text(n)
        symbol = str(n.get("symbol") or "")
        # roundRect -> rectangular, diamond -> decision
        if symbol == "diamond":
            # Decision node
            node_lines.append(f'  {nid}{{"{label}"}}')
        else:
            node_lines.append(f'  {nid}["{label}"]')

    # Links
    edge_lines: List[str] = []
    for l in links:
        s = _safe_id(str(l.get("source") or ""))
        t = _safe_id(str(l.get("target") or ""))
        lbl = ""
        try:
            lbl = (l.get("label") or {}).get("formatter") or ""
        except Exception:
            lbl = ""
        lbl = str(lbl).strip()
        lbl = lbl.replace('"', '\\"')
        if lbl:
            edge_lines.append(f'  {s} -- "{lbl}" --> {t}')
        else:
            edge_lines.append(f"  {s} --> {t}")

    # Mermaid init directive: tighter spacing + smoother edges + consistent theme.
    # Keeps layout compact and more readable in the chatbot panel.
    init = (
        '%%{init: {"flowchart":{"curve":"basis","nodeSpacing":18,"rankSpacing":28},'
        '"theme":"base","themeVariables":{"fontSize":"11px","primaryColor":"#2A7BE4","primaryTextColor":"#ffffff",'
        '"primaryBorderColor":"#1A5BBE","lineColor":"#6c757d","tertiaryColor":"#f8f9fa"}}}%%'
    )
    lines = [init, "graph LR"]
    lines.extend(node_lines)
    lines.extend(edge_lines)
    return "\n".join(lines)


def generate_agent_flow_diagram_from_state(
    agent_state: AgentState, agent_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate flow diagram data from agent state (pre-save).
    """
    if not getattr(agent_state, "rule_id", None):
        return None

    try:
        rule = get_rule(agent_state.rule_id)
        model = agent_state.fields.get("model") or rule.get("model", "yolov8n.pt")
        fps = agent_state.fields.get("fps") or rule.get("defaults", {}).get("fps", 1)

        rules = agent_state.fields.get("rules", [])
        if not rules and agent_state.rule_id:
            rules = [{"rule_id": agent_state.rule_id}]

        class TempAgent:
            def __init__(self):
                self.id = agent_id
                self.camera_id = agent_state.fields.get("camera_id", "")
                self.model = model
                self.fps = fps
                self.rules = rules
                self.run_mode = agent_state.fields.get("run_mode", "continuous")
                self.interval_minutes = agent_state.fields.get("interval_minutes")
                self.check_duration_seconds = agent_state.fields.get("check_duration_seconds")
                self.zone = agent_state.fields.get("zone")
                self.requires_zone = bool(agent_state.fields.get("zone"))

        return generate_agent_flow_diagram(TempAgent())
    except Exception:
        return None

