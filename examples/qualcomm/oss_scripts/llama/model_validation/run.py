#!/usr/bin/env python3
"""统一模型验证入口：工具直调，或按 JSON 配置运行分阶段消融流程。"""

import argparse
import subprocess
import sys
from pathlib import Path

from common.registry import REPO, TOOLS
from common.workflow import run_workflow


def main():
    if len(sys.argv) > 1 and sys.argv[1] in TOOLS:
        tool = TOOLS[sys.argv[1]]
        return subprocess.call(
            [sys.executable, str(tool.script), *sys.argv[2:]], cwd=REPO
        )
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="列出可调用的工具（不加载模型）")
    workflow = commands.add_parser(
        "workflow", help="展开配置并按阶段运行；默认不启动任务"
    )
    workflow.add_argument("--config", type=Path, required=True)
    selection = workflow.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stage", nargs="+", help="只运行指定阶段")
    selection.add_argument(
        "--all",
        action="store_true",
        help="明确运行配置内全部阶段，包括导出（若配置了）",
    )
    workflow.add_argument("--with-dependencies", action="store_true")
    workflow.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印完整命令，不创建文件或启动设备/GPU",
    )
    workflow.add_argument("--resume", action="store_true")
    workflow.add_argument(
        "--device-idle-confirmed",
        action="store_true",
        help="设备任务失败后，确认手机旧 driver 已退出再允许恢复",
    )
    args = parser.parse_args()
    try:
        if args.command == "list":
            for name, tool in TOOLS.items():
                print(f"{name:20s} {tool.description}")
        else:
            run_workflow(args)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
