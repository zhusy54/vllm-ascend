#!/usr/bin/env python3
"""Minimal Qwen3-14B smoke test for vllm-ascend on the current NPU."""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_PATH = os.environ.get(
    "QWEN3_14B_PATH",
    "/mnt/workspace/inductor/models/Qwen3-14B",
)


def main() -> int:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"model path missing: {model_path}", file=sys.stderr)
        return 2

    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "1+1等于几？只回答数字。"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=32, top_p=1.0),
    )
    text = outputs[0].outputs[0].text.strip()
    print("PROMPT:", prompt.replace("\n", "\\n")[:200])
    print("OUTPUT:", text)
    if not text:
        print("empty generation", file=sys.stderr)
        return 1
    print("Qwen3-14B smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
