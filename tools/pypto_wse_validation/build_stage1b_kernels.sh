#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: bash build_stage1b_kernels.sh OUTPUT_DIR" >&2
    exit 2
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is required" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
output_dir="$1"
t04_source="$script_dir/kernels/stage1b_t04.cpp"
t05_source="$script_dir/kernels/stage1b_t05.cpp"
t06_source="$script_dir/kernels/stage1b_t06.cpp"
t07_source="$script_dir/kernels/stage1b_t07.cpp"
t08_source="$script_dir/kernels/stage1b_t08.cpp"
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

"$ccec" "${common_flags[@]}" -DPYPTO_T04_DRIVER \
    -o "$output_dir/stage1b_t04_driver_vec.o" "$t04_source"
"$linker" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
    -o "$output_dir/stage1b_t04_driver.o" "$output_dir/stage1b_t04_driver_vec.o"

"$ccec" "${common_flags[@]}" -DPYPTO_T04_SERVICE \
    -o "$output_dir/stage1b_t04_service_vec.o" "$t04_source"
"$linker" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
    -o "$output_dir/stage1b_t04_service.o" "$output_dir/stage1b_t04_service_vec.o"

"$ccec" "${common_flags[@]}" -DPYPTO_T05_DRIVER \
    -o "$output_dir/stage1b_t05_driver_vec.o" "$t05_source"
"$linker" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
    -o "$output_dir/stage1b_t05_driver.o" "$output_dir/stage1b_t05_driver_vec.o"

"$ccec" "${common_flags[@]}" -DPYPTO_T05_SERVICE \
    -o "$output_dir/stage1b_t05_service_vec.o" "$t05_source"
"$linker" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
    -o "$output_dir/stage1b_t05_service.o" "$output_dir/stage1b_t05_service_vec.o"

"$ccec" "${common_flags[@]}" -DPYPTO_T06_DRIVER \
    -o "$output_dir/stage1b_t06_driver_vec.o" "$t06_source"
"$linker" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
    -o "$output_dir/stage1b_t06_driver.o" "$output_dir/stage1b_t06_driver_vec.o"

"$ccec" "${common_flags[@]}" -DPYPTO_T06_SERVICE \
    -o "$output_dir/stage1b_t06_service_vec.o" "$t06_source"
"$linker" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
    -o "$output_dir/stage1b_t06_service.o" "$output_dir/stage1b_t06_service_vec.o"

"$ccec" "${common_flags[@]}" -DPYPTO_T07_DRIVER \
    -o "$output_dir/stage1b_t07_driver_vec.o" "$t07_source"
"$linker" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
    -o "$output_dir/stage1b_t07_driver.o" "$output_dir/stage1b_t07_driver_vec.o"

"$ccec" "${common_flags[@]}" -DPYPTO_T07_SERVICE \
    -o "$output_dir/stage1b_t07_service_vec.o" "$t07_source"
"$linker" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
    -o "$output_dir/stage1b_t07_service.o" "$output_dir/stage1b_t07_service_vec.o"

"$ccec" "${common_flags[@]}" -DPYPTO_T08_DRIVER \
    -o "$output_dir/stage1b_t08_driver_vec.o" "$t08_source"
"$linker" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
    -o "$output_dir/stage1b_t08_driver.o" "$output_dir/stage1b_t08_driver_vec.o"

"$ccec" "${common_flags[@]}" -DPYPTO_T08_SERVICE \
    -o "$output_dir/stage1b_t08_service_vec.o" "$t08_source"
"$linker" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
    -o "$output_dir/stage1b_t08_service.o" "$output_dir/stage1b_t08_service_vec.o"

sha256sum \
    "$output_dir/stage1b_t04_driver.o" \
    "$output_dir/stage1b_t04_service.o" \
    "$output_dir/stage1b_t05_driver.o" \
    "$output_dir/stage1b_t05_service.o" \
    "$output_dir/stage1b_t06_driver.o" \
    "$output_dir/stage1b_t06_service.o" \
    "$output_dir/stage1b_t07_driver.o" \
    "$output_dir/stage1b_t07_service.o" \
    "$output_dir/stage1b_t08_driver.o" \
    "$output_dir/stage1b_t08_service.o"
