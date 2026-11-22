"""Public commands; heavyweight backends are imported only when selected."""

from __future__ import annotations
import argparse
import runpy
import sys

COMMANDS = {
    "run": "autosft.config.experiments",
    "round": "autosft.orchestration.standalone",
    "datasets": "autosft.data.download",
    "check": "autosft.orchestration.preflight",
}
TOOLS = {
    "train": "autosft.backends.megatron.train",
    "convert": "autosft.backends.megatron.convert",
    "evaluate": "autosft.benchmarks.submit",
    "evaluate-round": "autosft.benchmarks.evaluate_round",
    "jobs": "autosft.backends.local.lifecycle",
    "prepare": "autosft.data.preparation",
    "decontaminate": "autosft.data.decontamination",
    "inspect": "autosft.synthesis.inspection",
    "annotate": "autosft.data.action_analysis",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="autosft", description="AutoSFT research service"
    )
    parser.add_argument("command", choices=[*COMMANDS, "tools"])
    args = parser.parse_args(sys.argv[1:2])
    if args.command == "tools":
        tool_parser = argparse.ArgumentParser(
            prog="autosft tools", description="Individual pipeline and analysis tools"
        )
        tool_parser.add_argument("tool", choices=list(TOOLS))
        tool = tool_parser.parse_args(sys.argv[2:3]).tool
        module, arguments, name = TOOLS[tool], sys.argv[3:], f"tools {tool}"
    else:
        module, arguments, name = COMMANDS[args.command], sys.argv[2:], args.command
    sys.argv = [f"autosft {name}", *arguments]
    runpy.run_module(module, run_name="__main__")
    return 0
