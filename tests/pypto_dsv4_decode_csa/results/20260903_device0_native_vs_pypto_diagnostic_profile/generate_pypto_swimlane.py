#!/usr/bin/env python3
# ruff: noqa: E501
"""Render the measured PyPTO device timeline as a deterministic SVG."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class Event:
    name: str
    kind: str
    stream: int
    task: int
    start: Decimal
    stop: Decimal

    @property
    def duration(self) -> Decimal:
        return self.stop - self.start


@dataclass(frozen=True)
class Replay:
    execute: Event
    wait: Event
    dmas: tuple[Event, ...]
    aicpu: Event
    aicore: Event

    @property
    def envelope(self) -> Decimal:
        return self.wait.stop - self.execute.start


def _decimal(value: str) -> Decimal:
    return Decimal(value.strip())


def load_events(path: Path) -> list[Event]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Event(
            name=row["kernel_name"],
            kind=row["kernel_type"],
            stream=int(row["stream_id"]),
            task=int(row["task_id"]),
            start=_decimal(row["task_start(us)"]),
            stop=_decimal(row["task_stop(us)"]),
        )
        for row in rows
    ]


def select_pypto_replays(events: list[Event]) -> list[Replay]:
    executes = [event for event in events if event.kind == "MODEL_EXECUTE"]
    waits = [event for event in events if event.kind == "MODEL_WAIT_COMPLETE"]
    aicpus = [event for event in events if event.kind == "AI_CPU" and event.name.startswith("simpler_aicpu_l1_exec_")]
    aicores = [event for event in events if event.name == "aicore_kernel_0"]
    dmas = [event for event in events if event.kind == "MEMCPY_ASYNC"]

    replays: list[Replay] = []
    for aicpu in aicpus:
        execute = max((event for event in executes if event.start <= aicpu.start), key=lambda event: event.start)
        wait = min(
            (event for event in waits if event.start >= execute.start and event.stop >= aicpu.stop),
            key=lambda event: event.start,
        )
        aicore = next(
            event
            for event in aicores
            if event.start >= execute.start and event.stop <= wait.stop and event.start < aicpu.stop
        )
        replay_dmas = tuple(
            event for event in dmas if event.stream == aicpu.stream and execute.start <= event.start < aicpu.start
        )
        if len(replay_dmas) != 2:
            raise ValueError(f"expected two PyPTO DMA tasks, found {len(replay_dmas)}")
        replays.append(Replay(execute=execute, wait=wait, dmas=replay_dmas, aicpu=aicpu, aicore=aicore))

    if len(replays) != 2:
        raise ValueError(f"expected two profiled PyPTO replays, found {len(replays)}")
    return sorted(replays, key=lambda replay: replay.execute.start)


def render_svg(replays: list[Replay]) -> str:
    width = 1500
    height = 760
    left = 230
    right = 1440
    plot_width = right - left
    domain_us = Decimal("820")
    panel_tops = (105, 400)
    lane_offsets = (66, 126, 186)

    colors = {
        "ink": "#172033",
        "muted": "#64748b",
        "grid": "#dbe3ee",
        "panel": "#f8fafc",
        "model": "#94a3b8",
        "wait": "#e2e8f0",
        "dma1": "#f59e0b",
        "dma2": "#fb923c",
        "aicpu": "#c47a10",
        "aicore": "#2563eb",
        "white": "#ffffff",
    }

    def x(value: Decimal) -> float:
        return left + float(value / domain_us) * plot_width

    def fmt(value: Decimal) -> str:
        return f"{value:.3f}"

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">PyPTO ACLGraph 两轮 device 泳道图</title>',
        '<desc id="desc">两轮 PyPTO replay 的 MODEL control、AICPU stream 10 和 AICore stream 9 时间线，横轴为相对各轮 MODEL_EXECUTE 开始的微秒数。</desc>',
        "<defs>",
        '<pattern id="waitHatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">',
        f'<rect width="8" height="8" fill="{colors["wait"]}" opacity="0.55"/>',
        f'<line x1="0" y1="0" x2="0" y2="8" stroke="{colors["model"]}" stroke-width="2" opacity="0.45"/>',
        "</pattern>",
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="{colors["white"]}"/>',
        f'<text x="48" y="46" fill="{colors["ink"]}" font-family="Inter, Noto Sans CJK SC, Arial, sans-serif" font-size="25" font-weight="700">PyPTO ACLGraph device 泳道图</text>',
        f'<text x="48" y="75" fill="{colors["muted"]}" font-family="Inter, Noto Sans CJK SC, Arial, sans-serif" font-size="15">A3 device 0 · TRB · B4/S8 · C8191 · 统一横轴 0–820 us · 长度按 profiler 真实时长绘制</text>',
    ]

    for replay_index, (replay, top) in enumerate(zip(replays, panel_tops), start=1):
        base = replay.execute.start
        panel_bottom = top + 245
        out.append(
            f'<rect x="36" y="{top - 20}" width="1404" height="260" rx="12" fill="{colors["panel"]}" stroke="{colors["grid"]}"/>'
        )
        out.append(
            f'<text x="55" y="{top + 10}" fill="{colors["ink"]}" font-family="Inter, Noto Sans CJK SC, Arial, sans-serif" font-size="18" font-weight="700">PyPTO replay #{replay_index}</text>'
        )
        out.append(
            f'<text x="250" y="{top + 10}" fill="{colors["muted"]}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="14">MODEL envelope {fmt(replay.envelope)} us</text>'
        )

        for tick in range(0, 821, 100):
            tick_x = x(Decimal(tick))
            out.append(
                f'<line x1="{tick_x:.2f}" y1="{top + 28}" x2="{tick_x:.2f}" y2="{panel_bottom - 20}" stroke="{colors["grid"]}" stroke-width="1"/>'
            )
            out.append(
                f'<text x="{tick_x:.2f}" y="{panel_bottom + 3}" text-anchor="middle" fill="{colors["muted"]}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12">{tick}</text>'
            )
        out.append(
            f'<text x="{right}" y="{panel_bottom + 3}" text-anchor="end" fill="{colors["muted"]}" font-family="Inter, Noto Sans CJK SC, Arial, sans-serif" font-size="12">相对时间（us）</text>'
        )

        lane_names = ("Graph control · stream 44", "AICPU · stream 10", "AICore · stream 9")
        for name, offset in zip(lane_names, lane_offsets):
            lane_y = top + offset
            out.append(
                f'<text x="55" y="{lane_y + 20}" fill="{colors["ink"]}" font-family="Inter, Noto Sans CJK SC, Arial, sans-serif" font-size="14">{escape(name)}</text>'
            )
            out.append(
                f'<line x1="{left}" y1="{lane_y + 15}" x2="{right}" y2="{lane_y + 15}" stroke="{colors["grid"]}" stroke-width="1"/>'
            )

        control_y = top + lane_offsets[0]
        execute_start = replay.execute.start - base
        execute_width = max(2.0, float(replay.execute.duration / domain_us) * plot_width)
        wait_start = replay.wait.start - base
        wait_width = float(replay.wait.duration / domain_us) * plot_width
        out.append(
            f'<rect x="{x(execute_start):.2f}" y="{control_y}" width="{execute_width:.2f}" height="30" rx="3" fill="{colors["model"]}" stroke="{colors["muted"]}"/>'
        )
        out.append(
            f'<text x="{x(execute_start) + execute_width + 7:.2f}" y="{control_y - 7}" fill="{colors["muted"]}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12">MODEL_EXECUTE {fmt(replay.execute.duration)} us</text>'
        )
        out.append(
            f'<rect x="{x(wait_start):.2f}" y="{control_y}" width="{wait_width:.2f}" height="30" rx="3" fill="url(#waitHatch)" stroke="{colors["model"]}"/>'
        )
        out.append(
            f'<text x="{x(wait_start + replay.wait.duration / 2):.2f}" y="{control_y + 20}" text-anchor="middle" fill="{colors["ink"]}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="13">MODEL_WAIT_COMPLETE · {fmt(replay.wait.duration)} us</text>'
        )

        aicpu_y = top + lane_offsets[1]
        for dma_index, dma in enumerate(replay.dmas):
            dma_start = dma.start - base
            dma_width = max(2.0, float(dma.duration / domain_us) * plot_width)
            dma_color = colors[f"dma{dma_index + 1}"]
            out.append(
                f'<rect x="{x(dma_start):.2f}" y="{aicpu_y}" width="{dma_width:.2f}" height="30" rx="2" fill="{dma_color}" stroke="{colors["ink"]}" stroke-width="0.7"/>'
            )
        dma_text = " + ".join(f"DMA{index + 1} {fmt(dma.duration)}" for index, dma in enumerate(replay.dmas))
        out.append(
            f'<text x="{x(replay.dmas[0].start - base):.2f}" y="{aicpu_y - 7}" fill="{colors["muted"]}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12">{dma_text} us</text>'
        )

        aicpu_start = replay.aicpu.start - base
        aicpu_width = float(replay.aicpu.duration / domain_us) * plot_width
        out.append(
            f'<rect x="{x(aicpu_start):.2f}" y="{aicpu_y}" width="{aicpu_width:.2f}" height="30" rx="4" fill="{colors["aicpu"]}" stroke="#8a5209"/>'
        )
        out.append(
            f'<text x="{x(aicpu_start) + 12:.2f}" y="{aicpu_y + 20}" fill="{colors["white"]}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="13" font-weight="700">simpler_aicpu_l1_exec_* · {fmt(replay.aicpu.duration)} us</text>'
        )

        aicore_y = top + lane_offsets[2]
        aicore_start = replay.aicore.start - base
        aicore_width = float(replay.aicore.duration / domain_us) * plot_width
        out.append(
            f'<rect x="{x(aicore_start):.2f}" y="{aicore_y}" width="{aicore_width:.2f}" height="30" rx="4" fill="{colors["aicore"]}" stroke="#174ea6"/>'
        )
        out.append(
            f'<text x="{x(aicore_start) + 12:.2f}" y="{aicore_y + 20}" fill="{colors["white"]}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="13" font-weight="700">aicore_kernel_0 · {fmt(replay.aicore.duration)} us</text>'
        )

        lead = replay.aicore.start - replay.aicpu.start
        tail = replay.aicpu.stop - replay.aicore.stop
        tail_start = replay.aicore.stop - base
        tail_stop = replay.aicpu.stop - base
        annotation_y = aicore_y + 41
        out.extend(
            [
                f'<text x="{x(aicore_start) + 4:.2f}" y="{aicore_y - 7}" fill="{colors["muted"]}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12">AICPU lead {fmt(lead)} us</text>',
                f'<line x1="{x(tail_start):.2f}" y1="{annotation_y}" x2="{x(tail_stop):.2f}" y2="{annotation_y}" stroke="{colors["ink"]}" stroke-width="1.5"/>',
                f'<line x1="{x(tail_start):.2f}" y1="{annotation_y - 4}" x2="{x(tail_start):.2f}" y2="{annotation_y + 4}" stroke="{colors["ink"]}" stroke-width="1.5"/>',
                f'<line x1="{x(tail_stop):.2f}" y1="{annotation_y - 4}" x2="{x(tail_stop):.2f}" y2="{annotation_y + 4}" stroke="{colors["ink"]}" stroke-width="1.5"/>',
                f'<text x="{x((tail_start + tail_stop) / 2):.2f}" y="{annotation_y + 16}" text-anchor="middle" fill="{colors["ink"]}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12">tail {fmt(tail)} us</text>',
            ]
        )

    legend_y = 696
    legend = (
        (colors["model"], "MODEL_EXECUTE"),
        ("url(#waitHatch)", "MODEL_WAIT_COMPLETE"),
        (colors["dma1"], "MEMCPY_ASYNC"),
        (colors["aicpu"], "AICPU scheduler"),
        (colors["aicore"], "AICore kernel"),
    )
    legend_x = 55
    for fill, label in legend:
        out.append(
            f'<rect x="{legend_x}" y="{legend_y}" width="18" height="12" rx="2" fill="{fill}" stroke="{colors["muted"]}" stroke-width="0.6"/>'
        )
        out.append(
            f'<text x="{legend_x + 25}" y="{legend_y + 11}" fill="{colors["ink"]}" font-family="Inter, Noto Sans CJK SC, Arial, sans-serif" font-size="12">{escape(label)}</text>'
        )
        legend_x += 240

    out.append(
        f'<text x="55" y="738" fill="{colors["muted"]}" font-family="Inter, Noto Sans CJK SC, Arial, sans-serif" font-size="13">注：CANN profiler 不展开 aicore_kernel_0 内部 child task；本图只绘制可验证的外层 device 事件，不能解释为 child 算子全部串行。</text>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "profiler_output" / "task_time.csv")
    parser.add_argument("--output", type=Path, default=root / "pypto_device_swimlane.svg")
    args = parser.parse_args()

    replays = select_pypto_replays(load_events(args.input))
    args.output.write_text(render_svg(replays), encoding="utf-8")
    for index, replay in enumerate(replays, start=1):
        print(
            f"replay {index}: envelope={replay.envelope} us, "
            f"aicpu={replay.aicpu.duration} us, aicore={replay.aicore.duration} us, "
            f"tail={replay.aicpu.stop - replay.aicore.stop} us"
        )
    print(args.output)


if __name__ == "__main__":
    main()
