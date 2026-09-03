#!/usr/bin/env python3
"""Split the combined ABBA profiler trace into Native and PyPTO traces.

The source trace is kept intact.  Derived traces preserve the source timestamps and
event payloads so that either output can be opened directly by a trace viewer.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any

Event = dict[str, Any]
Window = tuple[Decimal, Decimal]


def _timestamp(event: Event) -> Decimal:
    return Decimal(str(event["ts"]))


def _end_timestamp(event: Event) -> Decimal:
    return _timestamp(event) + Decimal(str(event.get("dur", 0)))


def _model_id(event: Event) -> int | None:
    model_id = event.get("args", {}).get("Model Id")
    return model_id if isinstance(model_id, int) else None


def _process_names(events: Iterable[Event]) -> dict[int, str]:
    return {
        event["pid"]: event.get("args", {}).get("name", "")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name" and isinstance(event.get("pid"), int)
    }


def _discover_model_ids(events: list[Event], hardware_pid: int) -> tuple[int, int]:
    model_ids = {
        model_id
        for event in events
        if event.get("ph") == "X"
        and event.get("pid") == hardware_pid
        and (model_id := _model_id(event)) is not None
        and model_id != 0xFFFFFFFF
    }
    pypto_model_ids = {
        _model_id(event)
        for event in events
        if event.get("ph") == "X" and event.get("pid") == hardware_pid and event.get("name") == "aicore_kernel_0"
    }
    pypto_model_ids.discard(None)
    if len(pypto_model_ids) != 1:
        raise RuntimeError(
            f"expected one PyPTO Model Id identified by aicore_kernel_0, found {sorted(pypto_model_ids)}"
        )

    pypto_model_id = next(iter(pypto_model_ids))
    native_model_ids = model_ids - {pypto_model_id}
    if len(native_model_ids) != 1:
        raise RuntimeError(f"expected one Native Model Id beside the PyPTO Model Id, found {sorted(native_model_ids)}")
    return next(iter(native_model_ids)), pypto_model_id


def _model_windows(events: list[Event], hardware_pid: int, model_id: int) -> list[Window]:
    executes = sorted(
        (
            event
            for event in events
            if event.get("ph") == "X"
            and event.get("pid") == hardware_pid
            and event.get("name") == "MODEL_EXECUTE"
            and _model_id(event) == model_id
        ),
        key=_timestamp,
    )
    waits = sorted(
        (
            event
            for event in events
            if event.get("ph") == "X"
            and event.get("pid") == hardware_pid
            and event.get("name") == "MODEL_WAIT_COMPLETE"
            and _model_id(event) == model_id
        ),
        key=_timestamp,
    )
    if len(executes) != len(waits) or not executes:
        raise RuntimeError(f"unpaired MODEL events for Model Id {model_id}: {len(executes)} execute, {len(waits)} wait")

    windows = [(_timestamp(execute), _end_timestamp(wait)) for execute, wait in zip(executes, waits, strict=True)]
    if any(start >= end for start, end in windows):
        raise RuntimeError(f"invalid MODEL window for Model Id {model_id}")
    return windows


def _marker_windows(events: list[Event], marker_name: str) -> list[Window]:
    markers = sorted(
        (event for event in events if event.get("ph") == "X" and event.get("name") == marker_name),
        key=_timestamp,
    )
    if not markers:
        raise RuntimeError(f"missing host marker {marker_name!r}")
    return [(_timestamp(event), _end_timestamp(event)) for event in markers]


def _overlaps(event: Event, windows: Iterable[Window]) -> bool:
    start = _timestamp(event)
    end = _end_timestamp(event)
    return any(start <= window_end and end >= window_start for window_start, window_end in windows)


def _contained_by(event: Event, windows: Iterable[Window]) -> bool:
    start = _timestamp(event)
    end = _end_timestamp(event)
    return any(start >= window_start and end <= window_end for window_start, window_end in windows)


def _select_trace(
    events: list[Event],
    *,
    model_id: int,
    marker_name: str,
    process_names: dict[int, str],
    hardware_pid: int,
) -> list[Event]:
    model_windows = _model_windows(events, hardware_pid, model_id)
    marker_windows = _marker_windows(events, marker_name)
    device_timeline_pids = {
        pid for pid, name in process_names.items() if name in {"Ascend Hardware", "AI Core Freq", "Overlap Analysis"}
    }

    selected_ids: set[int] = set()
    for index, event in enumerate(events):
        phase = event.get("ph")
        if phase == "M" or "ts" not in event:
            continue
        # The source contains flow-end records without corresponding flow-start
        # records.  They add no lane duration and would be dangling after a split.
        if phase in {"s", "t", "f"}:
            continue

        pid = event.get("pid")
        if pid == hardware_pid:
            if _model_id(event) == model_id:
                selected_ids.add(index)
        elif pid in device_timeline_pids:
            if _overlaps(event, model_windows):
                selected_ids.add(index)
        elif _contained_by(event, marker_windows):
            # Keep the selected enqueue marker and its nested Python/CANN spans,
            # but omit process-wide enclosing spans such as ProfilerStep.
            selected_ids.add(index)

    selected_events = [events[index] for index in sorted(selected_ids)]
    selected_pids = {event.get("pid") for event in selected_events}
    selected_threads = {(event.get("pid"), event.get("tid")) for event in selected_events}

    def metadata_is_used(event: Event) -> bool:
        if event.get("ph") != "M":
            return False
        name = str(event.get("name", ""))
        if name.startswith("thread_"):
            return (event.get("pid"), event.get("tid")) in selected_threads
        return event.get("pid") in selected_pids

    return [event for index, event in enumerate(events) if index in selected_ids or metadata_is_used(event)]


def _validate(events: list[Event], *, mode: str, model_id: int, hardware_pid: int) -> None:
    hardware_events = [event for event in events if event.get("ph") == "X" and event.get("pid") == hardware_pid]
    unexpected_model_ids = {
        event_model_id
        for event in hardware_events
        if (event_model_id := _model_id(event)) is not None and event_model_id != model_id
    }
    if unexpected_model_ids:
        raise RuntimeError(f"{mode} trace contains unexpected Model Ids: {sorted(unexpected_model_ids)}")

    expected_marker = f"csa_steady_state.{mode}.enqueue"
    marker_count = sum(event.get("name") == expected_marker for event in events)
    execute_count = sum(event.get("name") == "MODEL_EXECUTE" for event in hardware_events)
    wait_count = sum(event.get("name") == "MODEL_WAIT_COMPLETE" for event in hardware_events)
    if (marker_count, execute_count, wait_count) != (2, 2, 2):
        raise RuntimeError(
            f"{mode} trace expected two host markers and two MODEL pairs, found "
            f"{marker_count}, {execute_count}, {wait_count}"
        )

    if mode == "pypto":
        aicpu_count = sum(str(event.get("name", "")).startswith("simpler_aicpu") for event in hardware_events)
        aicore_count = sum(event.get("name") == "aicore_kernel_0" for event in hardware_events)
        if (aicpu_count, aicore_count) != (2, 2):
            raise RuntimeError(
                f"PyPTO trace expected two AICPU and two AICore outer kernels, found {aicpu_count} and {aicore_count}"
            )


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=root / "profiler_output" / "trace_view.json")
    parser.add_argument("--output-dir", type=Path, default=root)
    args = parser.parse_args()

    with args.source.open(encoding="utf-8") as source_file:
        events: list[Event] = json.load(source_file)
    if not isinstance(events, list):
        raise RuntimeError("trace_view.json must contain a top-level event list")

    process_names = _process_names(events)
    hardware_pids = [pid for pid, name in process_names.items() if name == "Ascend Hardware"]
    if len(hardware_pids) != 1:
        raise RuntimeError(f"expected one Ascend Hardware process, found {hardware_pids}")
    hardware_pid = hardware_pids[0]
    native_model_id, pypto_model_id = _discover_model_ids(events, hardware_pid)

    outputs = {
        "native": (native_model_id, "csa_steady_state.native.enqueue"),
        "pypto": (pypto_model_id, "csa_steady_state.pypto.enqueue"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode, (model_id, marker_name) in outputs.items():
        selected = _select_trace(
            events,
            model_id=model_id,
            marker_name=marker_name,
            process_names=process_names,
            hardware_pid=hardware_pid,
        )
        _validate(selected, mode=mode, model_id=model_id, hardware_pid=hardware_pid)
        output = args.output_dir / f"{mode}_trace_view.json"
        with output.open("w", encoding="utf-8") as output_file:
            json.dump(selected, output_file, ensure_ascii=False, separators=(",", ":"))
            output_file.write("\n")
        print(f"{mode}: Model Id {model_id}, {len(selected)} events -> {output}")


if __name__ == "__main__":
    main()
