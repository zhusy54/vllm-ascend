#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Offline single-card generate through PyptoQwen3ForCausalLM."""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_PATH = os.environ.get("QWEN3_14B_PATH", "/mnt/workspace/inductor/models/Qwen3-14B")
INDUCTOR_ROOT = "/mnt/workspace/inductor"


def _sanitize_sys_path() -> None:
    # /mnt/workspace/inductor on PYTHONPATH makes `import vllm` hit the source tree.
    cleaned = []
    for entry in sys.path:
        if os.path.abspath(entry) == INDUCTOR_ROOT:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


def main() -> int:
    _sanitize_sys_path()
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"model path missing: {model_path}", file=sys.stderr)
        return 2

    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("PYPTO_LIB_ROOT", str(Path(INDUCTOR_ROOT) / "pypto-lib"))
    os.environ.setdefault("PTO_PLATFORM", "a2a3")
    os.environ.setdefault("QWEN3_PA_BLOCK_DIM", "20")

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
        max_model_len=1024,
        block_size=128,
        gpu_memory_utilization=0.50,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
        enable_prefix_caching=False,
        hf_overrides={"architectures": ["PyptoQwen3ForCausalLM"]},
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
    if "2" not in text:
        print("greedy text does not contain 2", file=sys.stderr)
        return 1
    print("PYPTO_QWEN3_14B_GENERATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
