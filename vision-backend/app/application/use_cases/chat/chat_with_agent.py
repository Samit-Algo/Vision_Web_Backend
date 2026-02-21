"""
Agent chat use case: LangChain vision-rule agent only.
Streams tokens and state updates; supports HITL (pending_approval, pending_zone_input).
"""

import asyncio
import json
import re
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from ...dto.chat_dto import (
    ChatMessageRequest,
    ChatMessageResponse,
    PendingApprovalSchema,
    ApprovalSummarySchema,
    PendingZoneInputSchema,
)
from ....agents.exceptions import ValidationError, get_user_message


def get_interrupt_value(result: dict) -> dict | None:
    """Return the first interrupt value dict from result, or None."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else None


def extract_pending_approval(result: dict) -> PendingApprovalSchema | None:
    value = get_interrupt_value(result)
    if not value or "action_requests" not in value:
        return None
    action_requests = value.get("action_requests") or []
    first_name = (action_requests[0].get("name") if action_requests else None) or ""
    if first_name == "request_zone_drawing":
        return None
    return PendingApprovalSchema(
        action_requests=action_requests,
        review_configs=value.get("review_configs", []),
        summary=ApprovalSummarySchema(
            rule_id=result.get("rule_id"),
            config=result.get("config") or {},
        ),
    )


def extract_pending_zone_input(result: dict) -> PendingZoneInputSchema | None:
    value = get_interrupt_value(result)
    if not value or "action_requests" not in value:
        return None
    action_requests = value.get("action_requests") or []
    first_name = (action_requests[0].get("name") if action_requests else None) or ""
    if first_name != "request_zone_drawing":
        return None
    config = result.get("config") or {}
    rule_id = result.get("rule_id") or ""
    camera_id = config.get("camera_id") or ""
    frame_snapshot_url = f"/api/v1/cameras/{camera_id}/snapshot" if camera_id else None
    zone_type = "polygon"
    if rule_id:
        try:
            from ....agents.latest_agent_chat.vision_rule.state import get_rule
            rule = get_rule(rule_id)
            if rule:
                zs = rule.get("zone_support") or {}
                zone_type = zs.get("zone_type", "polygon")
        except Exception:
            pass
    return PendingZoneInputSchema(
        camera_id=camera_id or None,
        frame_snapshot_url=frame_snapshot_url,
        zone_type=zone_type,
        rule_id=rule_id or None,
        config=config,
    )


def get_content_from_messages(messages: list) -> str:
    """Return the last text content from the messages list (assistant reply)."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
        if content and isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "") or ""
    return ""


def get_token_delta(chunk: Any) -> str:
    if chunk is None:
        return ""
    text = getattr(chunk, "content", None)
    if isinstance(text, str):
        return text
    if hasattr(chunk, "content_blocks") and chunk.content_blocks:
        for block in chunk.content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "") or ""
            if getattr(block, "type", None) == "text":
                return getattr(block, "text", "") or ""
    return getattr(chunk, "text", "") or ""


def sanitize_assistant_text(text: str) -> str:
    """Remove <question> tags from assistant text for display."""
    if not text:
        return text
    return re.sub(r"</?question>", "", text, flags=re.IGNORECASE).strip()


class ChatWithAgentUseCase:
    """Chat with the vision-rule LangChain agent (streaming + HITL)."""

    def __init__(
        self,
        agent_repository: Optional[Any] = None,
        camera_repository: Optional[Any] = None,
    ) -> None:
        pass

    async def execute(
        self,
        request: ChatMessageRequest,
        user_id: Optional[str] = None,
    ) -> ChatMessageResponse:
        if not user_id:
            raise ValidationError(
                "user_id is required for agent creation chat",
                user_message="Authentication required. Please sign in to create agents.",
            )
        if request.resume is not None:
            if not request.session_id:
                raise ValidationError(
                    "session_id is required when sending resume.",
                    user_message="Session is required to approve or reject. Please try again.",
                )
            session_id = request.session_id
        elif request.message is not None:
            session_id = request.session_id or str(uuid.uuid4())
        else:
            raise ValidationError(
                "Provide either message or resume.",
                user_message="Send a message or an approval decision.",
            )

        from ....agents.latest_agent_chat.vision_rule.agent import (
            get_vision_rule_agent,
            get_session_config,
        )
        from langgraph.types import Command

        agent = get_vision_rule_agent()
        config = get_session_config(session_id)
        if request.resume is not None:
            decisions = (request.resume or {}).get("decisions") or []
            if decisions and isinstance(decisions[0], dict) and "zone" in decisions[0]:
                config = dict(config)
                config.setdefault("configurable", {})["zone_data"] = decisions[0]["zone"]

        try:
            if request.resume is not None:
                result = await asyncio.to_thread(
                    lambda: agent.invoke(Command(resume=request.resume), config)
                )
            else:
                user_message = build_user_message(request)
                initial_state: Dict[str, Any] = {
                    "messages": [{"role": "user", "content": user_message}],
                    "user_id": user_id,
                    "session_id": session_id,
                }
                result = await asyncio.to_thread(
                    lambda: agent.invoke(initial_state, config)
                )

            pending_zone = extract_pending_zone_input(result)
            if pending_zone:
                return ChatMessageResponse(
                    response="Please draw the zone in the canvas below, then click Save.",
                    session_id=session_id,
                    status="success",
                    pending_zone_input=pending_zone,
                )
            pending = extract_pending_approval(result)
            if pending:
                return ChatMessageResponse(
                    response="Please approve or reject saving the agent configuration below.",
                    session_id=session_id,
                    status="success",
                    pending_approval=pending,
                )
            reply = get_content_from_messages(result.get("messages") or [])
            reply = sanitize_assistant_text(reply or "").strip() or "Done."
            return ChatMessageResponse(
                response=reply,
                session_id=session_id,
                status="success",
            )
        except Exception as error:
            return ChatMessageResponse(
                response=get_user_message(error),
                session_id=session_id,
                status="error",
            )

    async def stream_execute(
        self,
        *,
        request: ChatMessageRequest,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not user_id:
            raise ValidationError(
                "user_id is required for agent creation chat",
                user_message="Authentication required. Please sign in to create agents.",
            )
        if request.resume is not None:
            if not request.session_id:
                raise ValidationError(
                    "session_id is required when sending resume.",
                    user_message="Session is required to approve or reject.",
                )
            session_id = request.session_id
        elif request.message is not None:
            session_id = request.session_id or str(uuid.uuid4())
        else:
            raise ValidationError(
                "Provide either message or resume.",
                user_message="Send a message or an approval decision.",
            )

        from ....agents.latest_agent_chat.vision_rule.agent import (
            get_vision_rule_agent,
            get_session_config,
        )
        from langgraph.types import Command

        agent = get_vision_rule_agent()
        config = get_session_config(session_id)
        if request.resume is not None:
            decisions = (request.resume or {}).get("decisions") or []
            if decisions and isinstance(decisions[0], dict) and "zone" in decisions[0]:
                config = dict(config)
                config.setdefault("configurable", {})["zone_data"] = decisions[0]["zone"]

        yield {"event": "meta", "data": {"session_id": session_id}}

        try:
            if request.resume is not None:
                input_arg = Command(resume=request.resume)
            else:
                user_message = build_user_message(request)
                input_arg = {
                    "messages": [{"role": "user", "content": user_message}],
                    "user_id": user_id,
                    "session_id": session_id,
                }

            stream_mode_arg = ["messages", "updates"]
            state_so_far: Dict[str, Any] = {}
            pending_approval = None
            pending_zone = None
            interrupted = False

            try:
                stream = agent.astream(
                    input_arg,
                    config=config,
                    stream_mode=stream_mode_arg,
                )
            except (TypeError, AttributeError):
                stream = None

            if stream is None:
                result = await asyncio.to_thread(
                    lambda: agent.invoke(input_arg, config)
                )
                state_so_far = result
                if isinstance(input_arg, Command):
                    pass
                else:
                    pending_zone = extract_pending_zone_input(result)
                    pending_approval = extract_pending_approval(result)
                    interrupted = pending_zone is not None or pending_approval is not None
            else:
                async for mode, chunk in stream:
                    if mode == "messages":
                        token, _ = chunk if isinstance(chunk, (list, tuple)) else (chunk, {})
                        delta = get_token_delta(token)
                        if delta:
                            yield {"event": "token", "data": {"delta": delta}}
                    elif mode == "updates":
                        if not isinstance(chunk, dict):
                            continue
                        for node_name, node_data in chunk.items():
                            if node_name == "__interrupt__":
                                result = {**state_so_far, "__interrupt__": node_data}
                                pending_zone = extract_pending_zone_input(result)
                                pending_approval = extract_pending_approval(result)
                                interrupted = True
                                break
                            if isinstance(node_data, dict):
                                state_so_far.update(node_data)
                        if interrupted:
                            break

            if pending_zone:
                yield {"event": "pending_zone_input", "data": pending_zone.model_dump()}
            if pending_approval:
                yield {"event": "pending_approval", "data": pending_approval.model_dump()}

            if interrupted:
                reply = "Please draw the zone in the canvas below, then click Save." if pending_zone else "Please approve or reject saving the agent configuration below."
            else:
                reply = get_content_from_messages(state_so_far.get("messages") or [])
                reply = sanitize_assistant_text(reply or "").strip() or "Done."

            response = ChatMessageResponse(
                response=reply,
                session_id=session_id,
                status="success",
                pending_approval=pending_approval,
                pending_zone_input=pending_zone,
            )
            yield {"event": "done", "data": response.model_dump()}

        except Exception as error:
            yield {"event": "error", "data": {"message": get_user_message(error)}}
            yield {
                "event": "done",
                "data": ChatMessageResponse(
                    response=get_user_message(error),
                    session_id=session_id,
                    status="error",
                ).model_dump(),
            }


def build_user_message(request: ChatMessageRequest) -> str:
    """Build the user message string from request (message, zone_data, camera_id, video_path)."""
    parts = [request.message or ""]
    if request.zone_data:
        parts.append("Zone data: " + json.dumps(request.zone_data))
    if request.camera_id:
        parts.append("Camera ID: " + str(request.camera_id))
    if request.video_path:
        parts.append("(Using uploaded video: " + (request.video_path or "") + ")")
    return "\n\n".join(part for part in parts if part).strip() or "Continue."
