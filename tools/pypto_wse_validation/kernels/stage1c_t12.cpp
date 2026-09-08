// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// This file is a part of the vllm-ascend project.

// Stage 1C T12: stop admission, drain four accepted requests, return every
// credit, and acknowledge service stop before Host releases either window.
#include "cce_aicore_intrinsics.h"
#include <stdint.h>

namespace {
constexpr uint64_t kWaitTimeoutCycles = 20000000000ULL;
constexpr uint64_t kPatternMagic = 0x505950544f433132ULL;
constexpr uint64_t kByteRepeat = 0x0101010101010101ULL;
constexpr uint64_t kSlotStrideWords = (4ULL * 1024ULL) / sizeof(uint64_t);

struct MetaHead { uint16_t type; uint16_t length; };
struct KernelTypeMeta { MetaHead head; uint32_t kernel_type; };
struct MixCoreMeta { MetaHead head; uint16_t aic_ratio; uint16_t aiv_ratio; };
struct KernelMeta { KernelTypeMeta kernel_type; MixCoreMeta mix_core; };
#define T12_KERNEL_META(kernel_name)                                                                        \
    static const KernelMeta g_##kernel_name##_meta __attribute__((used, section(".ascend.meta." #kernel_name))) = { \
        {{1, sizeof(uint32_t)}, 5}, {{3, sizeof(uint32_t)}, 0, 1}}

struct alignas(64) SignalLine { uint64_t sequence; uint64_t reserved[7]; };
struct alignas(64) MessageLine {
    uint64_t generation; uint64_t sequence; uint64_t payload_bytes; uint64_t checksum;
    uint64_t head_marker; uint64_t tail_marker; uint64_t status; uint64_t slot;
};
struct alignas(64) SlotControl { SignalLine signal; MessageLine message; };
struct alignas(64) LifecycleLine {
    uint64_t stop_new; uint64_t post_stop_attempts; uint64_t post_stop_rejections;
    uint64_t drain_started; uint64_t drain_completed; uint64_t service_stopped;
    uint64_t credits_returned; uint64_t reserved;
};
struct alignas(64) ReportBlock {
    uint64_t accepted; uint64_t processed; uint64_t validation_errors; uint64_t sequence_errors;
    uint64_t generation_errors; uint64_t checksum_errors; uint64_t marker_errors; uint64_t timeouts;
    uint64_t elapsed_cycles; uint64_t submissions; uint64_t completions; uint64_t credits_acquired;
    uint64_t credits_returned; uint64_t terminal_tasks; uint64_t max_inflight;
    uint64_t stop_new_submissions; uint64_t post_stop_attempts; uint64_t post_stop_rejections;
    uint64_t drain_started; uint64_t drain_completed; uint64_t service_stopped;
    uint64_t input_fences; uint64_t output_fences; uint64_t reserved;
};
struct alignas(64) ControlBlock { SlotControl slots[2]; LifecycleLine lifecycle; ReportBlock report; };
struct Counters {
    uint64_t accepted; uint64_t processed; uint64_t validation_errors; uint64_t sequence_errors;
    uint64_t generation_errors; uint64_t checksum_errors; uint64_t marker_errors; uint64_t timeouts;
    uint64_t submissions; uint64_t completions; uint64_t credits_acquired; uint64_t credits_returned;
    uint64_t terminal_tasks; uint64_t input_fences; uint64_t output_fences;
};
static_assert(sizeof(SlotControl) == 128);
static_assert(sizeof(LifecycleLine) == 64);
static_assert(sizeof(ReportBlock) == 192);
static_assert(sizeof(ControlBlock) == 512);

__aicore__ inline void Store64(__gm__ uint64_t *address, uint64_t value) { __builtin_cce_st_dev(value, address, 0); }
__aicore__ inline uint64_t Load64(__gm__ uint64_t *address) {
    return static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
}
__aicore__ inline uint64_t PatternWord(uint64_t generation, uint64_t sequence, uint64_t index) {
    return kPatternMagic ^ (generation * 0x9e3779b97f4a7c15ULL) ^
        (sequence * 0xd1b54a32d192ed03ULL) ^ (index * 0x94d049bb133111ebULL);
}
__aicore__ inline __gm__ uint64_t *PayloadSlot(__gm__ uint64_t *base, uint64_t slot) {
    return base + slot * kSlotStrideWords;
}
__aicore__ inline bool Wait(__gm__ uint64_t *address, uint64_t expected) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    do { if (Load64(address) == expected) return true; }
    while (static_cast<uint64_t>(get_sys_cnt()) - begin < kWaitTimeoutCycles);
    return Load64(address) == expected;
}
__aicore__ inline void Publish(
    __gm__ SlotControl *control, uint64_t generation, uint64_t sequence, uint64_t checksum,
    uint64_t head, uint64_t tail, uint64_t status, uint64_t slot
) {
    Store64(&control->message.generation, generation); Store64(&control->message.sequence, sequence);
    Store64(&control->message.payload_bytes, 4ULL * 1024ULL); Store64(&control->message.checksum, checksum);
    Store64(&control->message.head_marker, head); Store64(&control->message.tail_marker, tail);
    Store64(&control->message.status, status); Store64(&control->message.slot, slot);
    dsb(DSB_ALL); Store64(&control->signal.sequence, sequence); dsb(DSB_ALL);
}
__aicore__ inline void Report(__gm__ ReportBlock *report, const Counters &c, uint64_t elapsed) {
    Store64(&report->accepted, c.accepted); Store64(&report->processed, c.processed);
    Store64(&report->validation_errors, c.validation_errors + c.sequence_errors + c.generation_errors +
        c.checksum_errors + c.marker_errors + c.timeouts);
    Store64(&report->sequence_errors, c.sequence_errors); Store64(&report->generation_errors, c.generation_errors);
    Store64(&report->checksum_errors, c.checksum_errors); Store64(&report->marker_errors, c.marker_errors);
    Store64(&report->timeouts, c.timeouts); Store64(&report->elapsed_cycles, elapsed);
    Store64(&report->submissions, c.submissions); Store64(&report->completions, c.completions);
    Store64(&report->credits_acquired, c.credits_acquired); Store64(&report->credits_returned, c.credits_returned);
    Store64(&report->terminal_tasks, c.terminal_tasks); Store64(&report->max_inflight, 2);
    Store64(&report->stop_new_submissions, 1); Store64(&report->post_stop_attempts, 1);
    Store64(&report->post_stop_rejections, 1); Store64(&report->drain_started, 1);
    Store64(&report->drain_completed, 1); Store64(&report->service_stopped, 1);
    Store64(&report->input_fences, c.input_fences); Store64(&report->output_fences, c.output_fences);
    Store64(&report->reserved, 0); dsb(DSB_ALL);
}
}  // namespace

#if defined(PYPTO_T12_DRIVER)
T12_KERNEL_META(pypto_stage1c_t12_driver_0_mix_aiv);
namespace {
__aicore__ inline void Submit(
    __gm__ uint64_t *remote_payload, __gm__ SlotControl *control,
    uint64_t generation, uint64_t sequence, uint64_t slot, Counters &c
) {
    __gm__ uint64_t *payload = PayloadSlot(remote_payload, slot); uint64_t checksum = 0;
    for (uint64_t index = 0; index < kSlotStrideWords; ++index) {
        const uint64_t value = PatternWord(generation, sequence, index); Store64(&payload[index], value); checksum += value;
    }
    dsb(DSB_ALL); Publish(control, generation, sequence, checksum, PatternWord(generation, sequence, 0),
        PatternWord(generation, sequence, kSlotStrideWords - 1), 0, slot);
    ++c.accepted; ++c.submissions; ++c.credits_acquired; ++c.input_fences;
}
__aicore__ inline bool Complete(
    __gm__ uint64_t *local_payload, __gm__ SlotControl *control,
    uint64_t generation, uint64_t sequence, uint64_t slot, Counters &c
) {
    if (!Wait(&control->signal.sequence, sequence)) { ++c.timeouts; return false; }
    dsb(DSB_ALL); __gm__ MessageLine *m = &control->message;
    if (Load64(&m->generation) != generation) ++c.generation_errors;
    if (Load64(&m->sequence) != sequence || Load64(&m->slot) != slot) ++c.sequence_errors;
    if (Load64(&m->payload_bytes) != 4ULL * 1024ULL || Load64(&m->status) != 0) ++c.validation_errors;
    uint64_t checksum = 0; uint64_t mismatches = 0; const uint64_t mask = (sequence & 0xffULL) * kByteRepeat;
    __gm__ uint64_t *payload = PayloadSlot(local_payload, slot);
    for (uint64_t index = 0; index < kSlotStrideWords; ++index) {
        const uint64_t value = Load64(&payload[index]); checksum += value;
        if (value != (PatternWord(generation, sequence, index) ^ mask)) ++mismatches;
    }
    if (mismatches || checksum != Load64(&m->checksum)) ++c.checksum_errors;
    if (Load64(&m->head_marker) != (PatternWord(generation, sequence, 0) ^ mask) ||
        Load64(&m->tail_marker) != (PatternWord(generation, sequence, kSlotStrideWords - 1) ^ mask))
        ++c.marker_errors;
    c.validation_errors += mismatches; ++c.completions; ++c.processed; ++c.credits_returned; ++c.terminal_tasks; return true;
}
}  // namespace
extern "C" __global__ __aicore__ void pypto_stage1c_t12_driver_0_mix_aiv(
    __gm__ uint64_t *local_payload, __gm__ ControlBlock *local_completion,
    __gm__ uint64_t *remote_payload, __gm__ ControlBlock *remote_submission,
    uint64_t generation
) {
    Counters c = {}; const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    for (uint64_t first = 1; first <= 4; first += 2) {
        Submit(remote_payload, &remote_submission->slots[0], generation, first, 0, c);
        Submit(remote_payload, &remote_submission->slots[1], generation, first + 1, 1, c);
        if (first == 3) {
            Store64(&remote_submission->lifecycle.stop_new, 1);
            Store64(&remote_submission->lifecycle.post_stop_attempts, 1);
            Store64(&remote_submission->lifecycle.post_stop_rejections, 1);
            Store64(&remote_submission->lifecycle.drain_started, 1); dsb(DSB_ALL);
        }
        if (!Complete(local_payload, &local_completion->slots[1], generation, first + 1, 1, c)) break;
        if (!Complete(local_payload, &local_completion->slots[0], generation, first, 0, c)) break;
    }
    if (!Wait(&local_completion->lifecycle.service_stopped, 1)) ++c.timeouts;
    Report(&local_completion->report, c, static_cast<uint64_t>(get_sys_cnt()) - begin);
}

#elif defined(PYPTO_T12_SERVICE)
T12_KERNEL_META(pypto_stage1c_t12_service_0_mix_aiv);
namespace {
__aicore__ inline bool Process(
    __gm__ uint64_t *local_payload, __gm__ SlotControl *submission,
    __gm__ uint64_t *remote_payload, __gm__ SlotControl *completion,
    uint64_t generation, uint64_t sequence, uint64_t slot, Counters &c
) {
    if (!Wait(&submission->signal.sequence, sequence)) { ++c.timeouts; return false; }
    dsb(DSB_ALL); __gm__ MessageLine *m = &submission->message;
    if (Load64(&m->generation) != generation) ++c.generation_errors;
    if (Load64(&m->sequence) != sequence || Load64(&m->slot) != slot) ++c.sequence_errors;
    if (Load64(&m->payload_bytes) != 4ULL * 1024ULL) ++c.validation_errors;
    __gm__ uint64_t *input = PayloadSlot(local_payload, slot); __gm__ uint64_t *output = PayloadSlot(remote_payload, slot);
    const uint64_t mask = (sequence & 0xffULL) * kByteRepeat; uint64_t in_sum = 0; uint64_t out_sum = 0; uint64_t mismatches = 0;
    for (uint64_t index = 0; index < kSlotStrideWords; ++index) {
        const uint64_t value = Load64(&input[index]); const uint64_t transformed = value ^ mask;
        if (value != PatternWord(generation, sequence, index)) ++mismatches;
        in_sum += value; out_sum += transformed; Store64(&output[index], transformed);
    }
    if (mismatches || in_sum != Load64(&m->checksum)) ++c.checksum_errors;
    if (Load64(&m->head_marker) != PatternWord(generation, sequence, 0) ||
        Load64(&m->tail_marker) != PatternWord(generation, sequence, kSlotStrideWords - 1))
        ++c.marker_errors;
    c.validation_errors += mismatches; dsb(DSB_ALL);
    Publish(completion, generation, sequence, out_sum, PatternWord(generation, sequence, 0) ^ mask,
        PatternWord(generation, sequence, kSlotStrideWords - 1) ^ mask,
        c.validation_errors + c.sequence_errors + c.generation_errors + c.checksum_errors, slot);
    ++c.accepted; ++c.submissions; ++c.completions; ++c.processed; ++c.terminal_tasks; ++c.output_fences; return true;
}
}  // namespace
extern "C" __global__ __aicore__ void pypto_stage1c_t12_service_0_mix_aiv(
    __gm__ uint64_t *local_payload, __gm__ ControlBlock *local_submission,
    __gm__ uint64_t *remote_payload, __gm__ ControlBlock *remote_completion,
    uint64_t generation
) {
    Counters c = {}; const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    for (uint64_t first = 1; first <= 4; first += 2) {
        if (!Process(local_payload, &local_submission->slots[1], remote_payload,
                     &remote_completion->slots[1], generation, first + 1, 1, c)) break;
        if (!Process(local_payload, &local_submission->slots[0], remote_payload,
                     &remote_completion->slots[0], generation, first, 0, c)) break;
    }
    if (Load64(&local_submission->lifecycle.stop_new) != 1 ||
        Load64(&local_submission->lifecycle.post_stop_attempts) != 1 ||
        Load64(&local_submission->lifecycle.post_stop_rejections) != 1 ||
        Load64(&local_submission->lifecycle.drain_started) != 1) ++c.validation_errors;
    Store64(&remote_completion->lifecycle.drain_completed, 1);
    Store64(&remote_completion->lifecycle.service_stopped, 1);
    Store64(&remote_completion->lifecycle.credits_returned, 4); dsb(DSB_ALL);
    Report(&local_submission->report, c, static_cast<uint64_t>(get_sys_cnt()) - begin);
}
#else
#error "Define PYPTO_T12_DRIVER or PYPTO_T12_SERVICE"
#endif
