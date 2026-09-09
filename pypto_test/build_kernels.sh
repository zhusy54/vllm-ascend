#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "usage: bash pypto_test/build_kernels.sh [OUTPUT_DIR]" >&2
    exit 2
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is required" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
output_dir="${1:-$script_dir/build}"
ccec="$ASCEND_HOME_PATH/bin/ccec"
linker="$ASCEND_HOME_PATH/bin/ld.lld"

mkdir -p "$output_dir"

common_flags=(
    -c
    -O3
    -g
    -x cce
    -Wall
    -std=c++17
    --cce-aicore-only
    --cce-aicore-arch=dav-c310-vec
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
)

for kernel in abc_driver b_service; do
    "$ccec" "${common_flags[@]}" \
        -o "$output_dir/${kernel}_vec.o" "$script_dir/pseudo_pypto/kernels/${kernel}.cpp"
    "$linker" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
        -o "$output_dir/${kernel}.o" "$output_dir/${kernel}_vec.o"
done

sha256sum "$output_dir/abc_driver.o" "$output_dir/b_service.o"
