#!/usr/bin/env python3
"""
blckhive agent CLI

Usage:
    python run_agent.py "Fix the failing tests"
    python run_agent.py --working-dir /my/project "Summarise the codebase"
    python run_agent.py --model claude-sonnet-4-6 "Quick question about this file"

Environment variables:
    ANTHROPIC_API_KEY   Required.  Your Anthropic API key.
    AGENT_MODEL         Override the default model.
    AGENT_MAX_TURNS     Override the default turn limit (30).
    AGENT_STATE_DIR     Override the state directory (~/.blckhive).
"""
import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="blckhive agentic assistant powered by Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task", nargs="?", help="Task to perform")
    parser.add_argument(
        "--working-dir", "-w",
        metavar="DIR",
        help="Working directory for file operations (default: cwd)",
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-opus-4-6",
        help="Model to use (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Maximum agentic loop turns (default: 30)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens per response (default: 8192)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress streaming output; print only the final answer",
    )
    parser.add_argument(
        "--skill", "-s",
        metavar="FILE",
        action="append",
        default=[],
        help="Load skills from a .py file (repeatable)",
    )
    parser.add_argument(
        "--list-mcp",
        action="store_true",
        help="List configured MCP servers and exit",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print execution summary after the task completes",
    )

    args = parser.parse_args()

    # Late import so --help works without anthropic installed
    from agent import create_agent, run

    agent = create_agent(
        model=args.model,
        working_dir=args.working_dir or Path.cwd(),
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        stream_output=not args.quiet,
    )

    # Load extra skill files
    for skill_path in args.skill:
        loaded = agent.skills.load_from_file(Path(skill_path))
        print(f"[cli] Loaded skills from {skill_path}: {', '.join(loaded) or 'none'}")

    # --list-mcp shortcut
    if args.list_mcp:
        servers = agent.mcp.list_servers()
        if not servers:
            print("No MCP servers configured.")
        else:
            print("Configured MCP servers:")
            for s in servers:
                state = "enabled" if s["enabled"] else "disabled"
                print(f"  [{state}] {s['name']}: {s['command']}")
                if s["description"]:
                    print(f"           {s['description']}")
        return

    # Require task
    if not args.task:
        # Interactive mode: read from stdin
        if sys.stdin.isatty():
            print("Enter task (Ctrl+D when done):")
        task = sys.stdin.read().strip()
        if not task:
            parser.print_help()
            sys.exit(1)
    else:
        task = args.task

    state = run(agent, task)

    if args.quiet:
        print(state.final_answer)
    elif args.summary:
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        print(state.summary)


if __name__ == "__main__":
    main()
