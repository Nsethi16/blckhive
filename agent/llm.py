"""
LLM client — wraps the Anthropic API with streaming and the agentic tool-use loop.

The loop runs until:
  - Claude calls `task_complete` (the state is marked done)
  - Claude returns stop_reason == "end_turn" without any tool calls
  - The turn limit is reached
  - An unrecoverable error occurs

Streaming is used throughout so partial output appears in real time.
Adaptive thinking is enabled on claude-opus-4-6 for complex reasoning.
"""
from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import anthropic

from .core import AgentState, Phase, ToolResult, SYSTEM_PROMPT

if TYPE_CHECKING:
    from .tools import ToolRegistry
    from .skills import SkillRegistry
    from .mcp import MCPManager
    from .config import AgentConfig


class LLMClient:
    """Drives the agentic loop using the Anthropic streaming API."""

    def __init__(self, config: "AgentConfig") -> None:
        self._config = config
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        state: AgentState,
        registry: "ToolRegistry",
        skills: "SkillRegistry",
        mcp: "MCPManager",
    ) -> None:
        """
        Drive the agentic loop until the task is complete or the turn limit is hit.
        Mutates `state` in place.
        """
        state.messages = [{"role": "user", "content": state.task}]

        # Build combined tool list: built-ins + skills + (MCP already injected)
        all_tools = registry.to_anthropic_tools() + skills.to_anthropic_tools()

        for turn in range(self._config.max_turns):
            if state.done:
                break

            # ---- Streaming call to Claude --------------------------------
            response_content, stop_reason = self._stream_turn(
                messages=state.messages,
                tools=all_tools,
            )

            # Append assistant turn
            state.messages.append({"role": "assistant", "content": response_content})

            # ---- Handle tool calls ---------------------------------------
            tool_use_blocks = [b for b in response_content if b.get("type") == "tool_use"]

            if not tool_use_blocks:
                # Claude finished without calling any tool
                text_blocks = [b for b in response_content if b.get("type") == "text"]
                state.final_answer = "\n".join(b.get("text", "") for b in text_blocks)
                state.done = True
                break

            # Execute every requested tool call
            tool_results = []
            for block in tool_use_blocks:
                tool_name: str = block["name"]
                arguments: dict = block.get("input", {})
                tool_use_id: str = block["id"]

                output, success = self._dispatch(
                    tool_name=tool_name,
                    arguments=arguments,
                    registry=registry,
                    skills=skills,
                    state=state,
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": output,
                        "is_error": not success,
                    }
                )

                # Check if the task was completed
                if tool_name == "task_complete":
                    state.final_answer = arguments.get("summary", output)
                    state.done = True

            # Feed results back
            state.messages.append({"role": "user", "content": tool_results})

            if state.done:
                break

        else:
            # Hit turn limit
            state.final_answer = (
                f"Turn limit ({self._config.max_turns}) reached. "
                "Task may be incomplete."
            )
            state.done = True

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _stream_turn(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[list[dict], str]:
        """
        Stream one Claude turn.  Returns (content_blocks, stop_reason).
        Prints text tokens to stdout in real time.
        """
        kwargs: dict = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "tools": tools,
        }

        # Adaptive thinking on Opus 4.6
        if "opus" in self._config.model:
            kwargs["thinking"] = {"type": "adaptive"}

        content_blocks: list[dict] = []
        current_block: dict | None = None
        stop_reason = "end_turn"

        with self._client.messages.stream(**kwargs) as stream:
            for event in stream:
                etype = event.type

                if etype == "content_block_start":
                    block = event.content_block
                    btype = block.type
                    if btype == "text":
                        current_block = {"type": "text", "text": ""}
                    elif btype == "tool_use":
                        current_block = {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": {},
                            "_input_json": "",
                        }
                    elif btype == "thinking":
                        current_block = {"type": "thinking", "thinking": ""}
                    else:
                        current_block = {"type": btype}

                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = delta.type
                    if dtype == "text_delta" and current_block:
                        text = delta.text
                        current_block["text"] = current_block.get("text", "") + text
                        if self._config.stream_output:
                            sys.stdout.write(text)
                            sys.stdout.flush()
                    elif dtype == "input_json_delta" and current_block:
                        current_block["_input_json"] = (
                            current_block.get("_input_json", "") + delta.partial_json
                        )
                    elif dtype == "thinking_delta" and current_block:
                        current_block["thinking"] = (
                            current_block.get("thinking", "") + delta.thinking
                        )

                elif etype == "content_block_stop":
                    if current_block is not None:
                        # Parse accumulated JSON for tool_use blocks
                        if current_block["type"] == "tool_use":
                            raw = current_block.pop("_input_json", "{}")
                            try:
                                current_block["input"] = json.loads(raw) if raw else {}
                            except json.JSONDecodeError:
                                current_block["input"] = {}
                        # Suppress empty thinking blocks
                        if current_block["type"] != "thinking" or current_block.get(
                            "thinking"
                        ):
                            content_blocks.append(current_block)
                        current_block = None

                elif etype == "message_delta":
                    stop_reason = getattr(event.delta, "stop_reason", "end_turn") or "end_turn"

        if self._config.stream_output:
            sys.stdout.write("\n")
            sys.stdout.flush()

        return content_blocks, stop_reason

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        tool_name: str,
        arguments: dict,
        registry: "ToolRegistry",
        skills: "SkillRegistry",
        state: AgentState,
    ) -> tuple[str, bool]:
        """Route a tool call to the right handler and record it in state."""
        if self._config.stream_output:
            print(f"\n[tool] {tool_name}({_fmt(arguments)})")

        # Determine phase from tool hint
        phase = Phase.ACT
        if registry.get(tool_name):
            tool_def = registry.get(tool_name)
            if tool_def and tool_def.phase_hint:
                phase = tool_def.phase_hint
        elif skills.get(tool_name):
            phase = Phase.ACT  # skills are actions

        state.phase = phase

        # Skill takes priority if same name registered in both
        if skills.get(tool_name):
            output, success = skills.execute(tool_name, arguments, registry)
        else:
            output, success = registry.execute(tool_name, arguments)

        if self._config.stream_output:
            short = output[:200].replace("\n", " ")
            status = "OK" if success else "FAIL"
            print(f"         -> [{status}] {short}")

        state.history.append(
            ToolResult(
                tool_name=tool_name,
                arguments=arguments,
                output=output,
                success=success,
                phase=phase,
            )
        )

        return output, success


def _fmt(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
