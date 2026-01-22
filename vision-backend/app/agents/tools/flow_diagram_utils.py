"""
Utility functions for generating agent processing flow diagrams.

Simplified top-to-bottom flow diagrams based on actual agent_main.py processing logic:
- Continuous mode: Camera → Initialize → Run Mode → Stop Check → YOLO → Rule Engine → Alert/Continue → Loop
- Patrol mode: Camera → Initialize → Run Mode → Sleep → Window Start → Window Check → Stop Check → YOLO → Rule Engine → Alert/Continue → Loop
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..session_state.agent_state import AgentState
from ...domain.models.agent import Agent
from .kb_utils import get_rule
import re


def generate_agent_flow_diagram(agent: Agent) -> Dict[str, Any]:
    """
    Generate professional top→bottom flow diagram for a saved agent.
    
    Shapes:
    - start/end: oval
    - process: rect
    - decision: diamond
    """
    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []

    def create_node(
        name: str,
        label: str,
        shape_type: str = "process",  # start, end, process, decision
        x: float = 0.0,
        y: float = 0.0,
    ) -> Dict[str, Any]:
        """Create a node with standardized shapes and styling."""
        
        # Map shape_type to symbol and sizes
        if shape_type == "start" or shape_type == "end":
            symbol = "oval"
            size = [140, 60]
            color = "#f8f9fa"
            border = "#6c757d"
        elif shape_type == "decision":
            symbol = "diamond"
            size = [180, 100]
            color = "#fefefe"
            border = "#dc3545" # Red for decisions
        else: # process
            symbol = "roundRect"
            size = [200, 70]
            color = "#ffffff"
            border = "#495057"

        return {
            "name": name,
            "symbol": symbol,
            "label": {
                "show": True,
                "formatter": label,
                "fontSize": 11,
                "color": "#212529",
                "position": "inside",
            },
            "symbolSize": size,
            "itemStyle": {
                "color": color,
                "borderColor": border,
                "borderWidth": 2,
                "borderRadius": 30 if symbol == "oval" else (8 if symbol == "roundRect" else 0),
            },
            "x": x,
            "y": y,
            "position": [x, y], # For Rete renderer
            "shape": symbol,     # For Rete renderer
        }

    # Extract agent properties
    fps = getattr(agent, "fps", None) or 1
    cam_id = (getattr(agent, "camera_id", "") or "").strip()
    cam_display = f"{cam_id[:12]}..." if len(cam_id) > 12 else (cam_id or "N/A")
    model_display = (
        getattr(agent, "model", None).replace(".pt", "")
        if getattr(agent, "model", None)
        else "YOLOv8"
    )

    # Get rule information
    rules = getattr(agent, "rules", None) or []
    rule_ids = [str(rc["rule_id"]) for rc in rules if isinstance(rc, dict) and rc.get("rule_id")]
    
    rule_label = "Condition Check"
    if len(rule_ids) == 1:
        try:
            rule_def = get_rule(rule_ids[0])
            rn = rule_def.get("rule_name", rule_ids[0])
            rule_label = f"Match Rule:\n{rn}"
        except Exception:
            rule_label = f"Match Rule:\n{rule_ids[0]}"
    elif len(rule_ids) > 1:
        rule_label = f"Check {len(rule_ids)} Rules"

    run_mode = getattr(agent, "run_mode", None) or "continuous"
    
    # Layout configuration
    x_center = 300.0
    y_start = 50.0
    y_gap = 120.0 # Vertical gap between nodes
    y = y_start

    # Build the Flow
    nodes.append(create_node("start", "START", "start", x=x_center, y=y))
    y += y_gap

    nodes.append(create_node("init", f"Initialize Camera\n{cam_display}", "process", x=x_center, y=y))
    y += y_gap

    if run_mode == "continuous":
        # Monitoring Loop
        nodes.append(create_node("stop_check", "User Stop?", "decision", x=x_center, y=y))
        y += y_gap
        
        nodes.append(create_node("process_frame", f"AI Processing\n{model_display}", "process", x=x_center, y=y))
        y += y_gap

        nodes.append(create_node("rule_check", rule_label, "decision", x=x_center, y=y))
        
        # Branches below decision
        y_branch = y + y_gap
        nodes.append(create_node("alert", "TRIGGER ALERT", "process", x=150.0, y=y_branch))
        nodes.append(create_node("loop_cont", "Continue Loop", "process", x=450.0, y=y_branch))
        
        # End node at the bottom
        nodes.append(create_node("end", "STOPPED", "end", x=x_center, y=y_branch + y_gap))

        # Links
        links.append({"source": "start", "target": "init"})
        links.append({"source": "init", "target": "stop_check"})
        
        links.append({
            "source": "stop_check", 
            "target": "process_frame", 
            "label": {"show": True, "formatter": "NO"}
        })
        links.append({
            "source": "stop_check", 
            "target": "end", 
            "label": {"show": True, "formatter": "YES"},
            "isExit": True
        })
        
        links.append({"source": "process_frame", "target": "rule_check"})
        
        links.append({
            "source": "rule_check", 
            "target": "alert", 
            "label": {"show": True, "formatter": "MATCH"}
        })
        links.append({
            "source": "rule_check", 
            "target": "loop_cont", 
            "label": {"show": True, "formatter": "SKIP"}
        })
        
        # Loop back to stop check
        links.append({
            "source": "alert", 
            "target": "stop_check", 
            "isLoop": True
        })
        links.append({
            "source": "loop_cont", 
            "target": "stop_check", 
            "isLoop": True
        })

    else:
        # Patrol Mode
        interval = getattr(agent, "interval_minutes", 5)
        nodes.append(create_node("sleep", f"Patrol Wait\n{interval} min", "process", x=x_center, y=y))
        y += y_gap

        nodes.append(create_node("window_check", "Inside Schedule?", "decision", x=x_center, y=y))
        y += y_gap

        nodes.append(create_node("stop_check", "User Stop?", "decision", x=x_center, y=y))
        y += y_gap

        nodes.append(create_node("process_frame", f"AI Processing\n{model_display}", "process", x=x_center, y=y))
        y += y_gap

        nodes.append(create_node("rule_check", rule_label, "decision", x=x_center, y=y))
        
        y_branch = y + y_gap
        # Branches
        nodes.append(create_node("alert", "TRIGGER ALERT", "process", x=150.0, y=y_branch))
        nodes.append(create_node("loop_cont", "Continue Patrol", "process", x=450.0, y=y_branch))
        
        # End node
        nodes.append(create_node("end", "STOPPED", "end", x=x_center, y=y_branch + y_gap))

        # Links for Patrol
        links.append({"source": "start", "target": "init"})
        links.append({"source": "init", "target": "sleep"})
        links.append({"source": "sleep", "target": "window_check"})
        
        links.append({
            "source": "window_check", 
            "target": "stop_check", 
            "label": {"show": True, "formatter": "YES"}
        })
        links.append({
            "source": "window_check", 
            "target": "sleep", 
            "label": {"show": True, "formatter": "NO"},
            "isLoop": True
        })
        
        links.append({
            "source": "stop_check", 
            "target": "process_frame", 
            "label": {"show": True, "formatter": "NO"}
        })
        links.append({
            "source": "stop_check", 
            "target": "end", 
            "label": {"show": True, "formatter": "YES"},
            "isExit": True
        })
        
        links.append({"source": "process_frame", "target": "rule_check"})
        links.append({
            "source": "rule_check", 
            "target": "alert", 
            "label": {"show": True, "formatter": "MATCH"}
        })
        links.append({
            "source": "rule_check", 
            "target": "loop_cont", 
            "label": {"show": True, "formatter": "SKIP"}
        })
        
        links.append({
            "source": "alert", 
            "target": "window_check", 
            "isLoop": True
        })
        links.append({
            "source": "loop_cont", 
            "target": "window_check", 
            "isLoop": True
        })


    return {
        "nodes": nodes,
        "links": links,
        "layout": "vertical",
        "direction": "top-to-bottom"
    }


    return {
        "nodes": nodes,
        "links": links,
        "layout": "vertical",
        "direction": "top-to-bottom"
    }


def generate_agent_flow_mermaid(agent: Agent) -> str:
    """
    Generate a Mermaid flow diagram (graph TD - top-down) for a saved agent.

    Output is text-only Mermaid syntax suitable for fenced blocks:
    ```mermaid
    graph TD
      ...
    ```

    This maps 1:1 to generate_agent_flow_diagram(...) output (nodes + links),
    but rendered in Mermaid instead of Rete.js.
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

    # Mermaid init directive: vertical layout with neutral colors
    init = (
        '%%{init: {"flowchart":{"curve":"basis","nodeSpacing":20,"rankSpacing":100},'
        '"theme":"base","themeVariables":{"fontSize":"11px","primaryColor":"#f5f5f5","primaryTextColor":"#333",'
        '"primaryBorderColor":"#666","lineColor":"#999","tertiaryColor":"#f8f9fa"}}}%%'
    )
    lines = [init, "graph TD"]
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
