#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: bash build_stage1c_kernels.sh OUTPUT_DIR" >&2
    exit 2
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is required" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
output_dir="$1"
ccec="$ASCEND_HOME_PATH/bin/ccec"
linker="$ASCEND_HOME_PATH/bin/ld.lld"
mkdir -p "$output_dir"

common_flags=(
    -c -O3 -g -x cce -Wall -std=c++17
    --cce-aicore-only --cce-aicore-arch=dav-c310-vec
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
)

build_kernel() {
    local define="$1"
    local source="$2"
    local output="$3"
    "$ccec" "${common_flags[@]}" "-D$define" -o "$output_dir/${output}_vec.o" "$source"
    "$linker" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
        -o "$output_dir/${output}.o" "$output_dir/${output}_vec.o"
}

build_kernel PYPTO_T09_DRIVER "$script_dir/kernels/stage1c_t09.cpp" stage1c_t09_driver
build_kernel PYPTO_T09_SERVICE "$script_dir/kernels/stage1c_t09.cpp" stage1c_t09_service
build_kernel PYPTO_FAULT_DRIVER "$script_dir/kernels/stage1c_fault.cpp" stage1c_fault_driver
build_kernel PYPTO_FAULT_SERVICE "$script_dir/kernels/stage1c_fault.cpp" stage1c_fault_service
build_kernel PYPTO_T12_DRIVER "$script_dir/kernels/stage1c_t12.cpp" stage1c_t12_driver
build_kernel PYPTO_T12_SERVICE "$script_dir/kernels/stage1c_t12.cpp" stage1c_t12_service

sha256sum \
    "$output_dir/stage1c_t09_driver.o" \
    "$output_dir/stage1c_t09_service.o" \
    "$output_dir/stage1c_fault_driver.o" \
    "$output_dir/stage1c_fault_service.o" \
    "$output_dir/stage1c_t12_driver.o" \
    "$output_dir/stage1c_t12_service.o"
