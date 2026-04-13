from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo


DEFAULT_LOG_PATHS = [
    Path("/Users/mingxiaoli/.codex/sessions/2026/04/12/rollout-2026-04-12T13-04-12-019d8013-74ec-7e32-8392-f64f5186a306.jsonl"),
    Path("/Users/mingxiaoli/.codex/archived_sessions/rollout-2026-04-12T13-04-12-019d8013-74ec-7e32-8392-f64f5186a306.jsonl"),
]
DEFAULT_TARGET = "/Users/mingxiaoli/Documents/AISpeech/realtime_web_demo.py"
KNOWN_SNAPSHOTS = {
    "realtime_ok_baseline": "call_UOXf7t17VchNAu0GY4HZYpbl",
    "realtime_initial": "call_8T8xhBRXqmKWfHeDT4mVpV1S",
    "realtime_streaming_20260412_2223": "call_NRYwezWGZnVJSUezCKNPex9b",
    "realtime_streaming_20260412_2241": "call_bMcVR9dtiSZ3esbuD7LNIEs0",
    "realtime_streaming_20260412_2246": "call_r31HKPjqbqIVPv77UGwp9nl3",
    "realtime_streaming_20260412_2258_asr8bit": "call_7Wm6WiGuvS1ADHuI50h5NLjt",
}
HUNK_RE = re.compile(r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@$")
DEFAULT_TIMEZONE = "Asia/Shanghai"


@dataclass
class Snapshot:
    source_log: Path
    line_no: int
    timestamp: str
    call_id: str
    change_type: str
    content: Optional[str] = None
    unified_diff: Optional[str] = None


def iter_snapshots(log_paths: Iterable[Path], target_file: str) -> List[Snapshot]:
    snapshots: List[Snapshot] = []
    seen = set()
    for log_path in log_paths:
        if not log_path.exists():
            continue
        with log_path.open() as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = obj.get("payload") or {}
                if payload.get("type") != "patch_apply_end":
                    continue
                changes = payload.get("changes") or {}
                change = changes.get(target_file)
                if not change:
                    continue
                key = (payload.get("call_id"), change.get("type"), obj.get("timestamp"))
                if key in seen:
                    continue
                seen.add(key)
                snapshots.append(
                    Snapshot(
                        source_log=log_path,
                        line_no=line_no,
                        timestamp=obj.get("timestamp", ""),
                        call_id=payload.get("call_id", ""),
                        change_type=change.get("type", ""),
                        content=change.get("content"),
                        unified_diff=change.get("unified_diff"),
                    )
                )
    snapshots.sort(key=lambda item: item.timestamp)
    return snapshots


def format_timestamp(timestamp: str, timezone_name: str) -> str:
    if not timestamp:
        return ""
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    local_dt = dt.astimezone(ZoneInfo(timezone_name))
    return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def _normalize_line(text: str) -> str:
    return text.rstrip("\n")


def apply_unified_diff(original: str, unified_diff: str) -> str:
    source_lines = original.splitlines(keepends=True)
    output_lines: List[str] = []
    source_index = 0
    diff_lines = unified_diff.splitlines()
    index = 0

    while index < len(diff_lines):
        line = diff_lines[index]
        if not line.startswith("@@"):
            index += 1
            continue

        match = HUNK_RE.match(line)
        if not match:
            raise ValueError(f"unsupported hunk header: {line}")

        old_start = int(match.group("old_start"))
        copy_until = max(0, old_start - 1)
        output_lines.extend(source_lines[source_index:copy_until])
        source_index = copy_until
        index += 1

        while index < len(diff_lines) and not diff_lines[index].startswith("@@"):
            hunk_line = diff_lines[index]
            if hunk_line == r"\ No newline at end of file":
                index += 1
                continue
            if not hunk_line:
                raise ValueError("empty diff line inside hunk")

            prefix = hunk_line[0]
            body = hunk_line[1:]

            if prefix == " ":
                if source_index >= len(source_lines):
                    raise ValueError("context line beyond end of source")
                if _normalize_line(source_lines[source_index]) != body:
                    raise ValueError(
                        f"context mismatch at source line {source_index + 1}: "
                        f"expected {body!r}, got {_normalize_line(source_lines[source_index])!r}"
                    )
                output_lines.append(source_lines[source_index])
                source_index += 1
            elif prefix == "-":
                if source_index >= len(source_lines):
                    raise ValueError("delete line beyond end of source")
                if _normalize_line(source_lines[source_index]) != body:
                    raise ValueError(
                        f"delete mismatch at source line {source_index + 1}: "
                        f"expected {body!r}, got {_normalize_line(source_lines[source_index])!r}"
                    )
                source_index += 1
            elif prefix == "+":
                output_lines.append(body + "\n")
            else:
                raise ValueError(f"unsupported diff prefix: {prefix!r}")
            index += 1

    output_lines.extend(source_lines[source_index:])
    return "".join(output_lines)


def materialize_snapshot(snapshots: List[Snapshot], stop_call_id: Optional[str] = None, stop_index: Optional[int] = None) -> str:
    content = ""
    applied = False
    for index, snapshot in enumerate(snapshots, 1):
        if snapshot.change_type == "add":
            content = snapshot.content or ""
        elif snapshot.change_type == "update":
            content = apply_unified_diff(content, snapshot.unified_diff or "")
        elif snapshot.change_type == "delete":
            content = ""
        else:
            raise ValueError(f"unsupported change type: {snapshot.change_type}")

        applied = True
        if stop_call_id and snapshot.call_id == stop_call_id:
            return content
        if stop_index is not None and index == stop_index:
            return content

    if stop_call_id:
        raise ValueError(f"call_id not found in snapshot chain: {stop_call_id}")
    if stop_index is not None:
        raise ValueError(f"index out of range: {stop_index}")
    return content if applied else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild historical demo snapshots from Codex session logs")
    parser.add_argument("--target-file", default=DEFAULT_TARGET, help="Absolute path of the historical file to reconstruct")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE, help="Timezone used when printing snapshot timestamps")
    parser.add_argument(
        "--log",
        action="append",
        dest="logs",
        help="JSONL session log path. Can be repeated. Defaults to the known AISpeech session logs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available snapshots for the target file")

    export_parser = subparsers.add_parser("export", help="Export a reconstructed snapshot to a file")
    export_parser.add_argument("--call-id", help="Stop after applying the snapshot with this call id")
    export_parser.add_argument("--index", type=int, help="Stop after applying the Nth snapshot")
    export_parser.add_argument("--snapshot-name", choices=sorted(KNOWN_SNAPSHOTS), help="Named snapshot alias")
    export_parser.add_argument("--output", required=True, help="Path to write the reconstructed file")
    export_parser.add_argument(
        "--header",
        action="store_true",
        help="Prepend a short comment noting that the file was reconstructed from Codex history",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logs = [Path(item) for item in (args.logs or [])] or DEFAULT_LOG_PATHS
    snapshots = iter_snapshots(logs, args.target_file)
    if not snapshots:
        raise SystemExit(f"no snapshots found for {args.target_file}")

    if args.command == "list":
        for idx, snapshot in enumerate(snapshots, 1):
            note = ""
            for name, call_id in KNOWN_SNAPSHOTS.items():
                if snapshot.call_id == call_id:
                    note = f" [{name}]"
                    break
            local_time = format_timestamp(snapshot.timestamp, args.timezone)
            print(
                f"{idx:02d}  {local_time}  {snapshot.change_type:6s}  "
                f"{snapshot.call_id}  {snapshot.source_log.name}:{snapshot.line_no}{note}"
            )
        return

    stop_call_id = args.call_id
    if args.snapshot_name:
        stop_call_id = KNOWN_SNAPSHOTS[args.snapshot_name]
    if not stop_call_id and args.index is None:
        raise SystemExit("export requires --call-id, --index, or --snapshot-name")

    content = materialize_snapshot(snapshots, stop_call_id=stop_call_id, stop_index=args.index)
    if args.header:
        marker = stop_call_id or f"index {args.index}"
        header = (
            f"# Reconstructed from Codex session history for {Path(args.target_file).name}.\n"
            f"# Snapshot: {marker}\n\n"
        )
        content = header + content

    output_path = Path(args.output)
    output_path.write_text(content)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
