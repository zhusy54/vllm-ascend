// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// This file is a part of the vllm-ascend project.

// Stage 1C T10/T11: one accepted request remains incomplete until a bounded
// device-side timeout marks the generation unhealthy and quarantines the slot.
#include "cce_aicore_intrinsics.h"
#include <stdint.h>

namespace {

constexpr uint64_t kPatternMagic = 0x505950544f434641ULL;
constexpr uint64_t kPausedStatus = 0x504155534544ULL;

struct MetaHead { uint16_t type; uint16_t length; };
struct KernelTypeMeta { MetaHead head; uint32_t kernel_type; };
struct MixCoreMeta { MetaHead head; uint16_t aic_ratio; uint16_t aiv_ratio; };
struct KernelMeta { KernelTypeMeta kernel_type; MixCoreMeta mix_core; };
#define FAULT_KERNEL_META(kernel_name)                                                                      \
    static const KernelMeta g_##kernel_name##_meta __attribute__((used, section(".ascend.meta." #kernel_name))) = { \
        {{1, sizeof(uint32_t)}, 5}, {{3, sizeof(uint32_t)}, 0, 1}}

struct alignas(64) SignalLine { uint64_t sequence; uint64_t reserved[7]; };
struct alignas(64) MessageLine {
    uint64_t generation;
    uint64_t sequence;
    uint64_t payload_bytes;
    uint64_t checksum;
    uint64_t head_marker;
    uint64_t tail_marker;
    uint64_t status;
    uint64_t slot;
};
struct alignas(64) SlotControl { SignalLine signal; MessageLine message; };
struct alignas(64) ReportBlock {
    uint64_t accepted;
    uint64_t timed_out;
    uint64_t successes;
    uint64_t validation_errors;
    uint64_t sequence_errors;
    uint64_t generation_errors;
    uint64_t timeouts;
    uint64_t elapsed_cycles;
    uint64_t submissions;
    uint64_t completions;
    uint64_t service_pause_observed;
    uint64_t generation_unhealthy;
    uint64_t accepting_new_requests;
    uint64_t quarantined_slots;
    uint64_t terminal_errors;
    uint64_t forged_success_completions;
    uint64_t credits_returned;
    uint64_t slot_reuse_after_timeout;
    uint64_t configured_timeout_cycles;
    uint64_t timeout_within_bound;
    uint64_t input_fences;
    uint64_t output_fences;
    uint64_t mode;
    uint64_t reserved;
};
struct alignas(64) ControlBlock { SlotControl slot; ReportBlock report; };

static_assert(sizeof(SlotControl) == 128);
static_assert(sizeof(ReportBlock) == 192);
static_assert(sizeof(ControlBlock) == 320);

__aicore__ inline void Store64(__gm__ uint64_t *address, uint64_t value) {
    __builtin_cce_st_dev(value, address, 0);
}
__aicore__ inline uint64_t Load64(__gm__ uint64_t *address) {
    return static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
}
__aicore__ inline uint64_t PatternWord(uint64_t generation, uint64_t index) {
    return kPatternMagic ^ (generation * 0x9e3779b97f4a7c15ULL) ^ (index * 0x94d049bb133111ebULL);
}
__attribute__((unused)) __aicore__ inline void PublishSubmission(
    __gm__ SlotControl *control, uint64_t generation, uint64_t words, uint64_t checksum
) {
    Store64(&control->message.generation, generation);
    Store64(&control->message.sequence, 1);
    Store64(&control->message.payload_bytes, words * sizeof(uint64_t));
    Store64(&control->message.checksum, checksum);
    Store64(&control->message.head_marker, PatternWord(generation, 0));
    Store64(&control->message.tail_marker, PatternWord(generation, words - 1));
    Store64(&control->message.status, 0);
    Store64(&control->message.slot, 0);
    dsb(DSB_ALL);
    Store64(&control->signal.sequence, 1);
    dsb(DSB_ALL);
}
__aicore__ inline void PublishReport(
    __gm__ ReportBlock *report, uint64_t elapsed, uint64_t timeout_cycles, uint64_t mode,
    uint64_t input_fences
) {
    Store64(&report->accepted, 1);
    Store64(&report->timed_out, 1);
    Store64(&report->successes, 0);
    Store64(&report->validation_errors, 0);
    Store64(&report->sequence_errors, 0);
    Store64(&report->generation_errors, 0);
    Store64(&report->timeouts, 1);
    Store64(&report->elapsed_cycles, elapsed);
    Store64(&report->submissions, 1);
    Store64(&report->completions, 0);
    Store64(&report->service_pause_observed, 1);
    Store64(&report->generation_unhealthy, 1);
    Store64(&report->accepting_new_requests, 0);
    Store64(&report->quarantined_slots, 1);
    Store64(&report->terminal_errors, 1);
    Store64(&report->forged_success_completions, 0);
    Store64(&report->credits_returned, 0);
    Store64(&report->slot_reuse_after_timeout, 0);
    Store64(&report->configured_timeout_cycles, timeout_cycles);
    Store64(&report->timeout_within_bound, elapsed >= timeout_cycles ? 1 : 0);
    Store64(&report->input_fences, input_fences);
    Store64(&report->output_fences, 0);
    Store64(&report->mode, mode);
    Store64(&report->reserved, 0);
    dsb(DSB_ALL);
}

}  // namespace

#if defined(PYPTO_FAULT_DRIVER)

FAULT_KERNEL_META(pypto_stage1c_fault_driver_0_mix_aiv);

extern "C" __global__ __aicore__ void pypto_stage1c_fault_driver_0_mix_aiv(
    __gm__ uint64_t *local_payload, __gm__ ControlBlock *local_completion,
    __gm__ uint64_t *remote_payload, __gm__ ControlBlock *remote_submission,
    uint64_t generation, uint64_t timeout_cycles, uint32_t payload_words, uint32_t mode
) {
    (void)local_payload;
    uint64_t checksum = 0;
    for (uint64_t index = 0; index < payload_words; ++index) {
        const uint64_t value = PatternWord(generation, index);
        Store64(&remote_payload[index], value);
        checksum += value;
    }
    dsb(DSB_ALL);
    PublishSubmission(&remote_submission->slot, generation, payload_words, checksum);
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint64_t pause_observed = 0;
    while (static_cast<uint64_t>(get_sys_cnt()) - begin < timeout_cycles) {
        if (Load64(&local_completion->slot.message.status) == kPausedStatus) pause_observed = 1;
        if (Load64(&local_completion->slot.signal.sequence) != 0) break;
    }
    const uint64_t elapsed = static_cast<uint64_t>(get_sys_cnt()) - begin;
    PublishReport(&local_completion->report, elapsed, timeout_cycles, mode, 1);
    Store64(&local_completion->report.service_pause_observed, pause_observed);
    Store64(
        &local_completion->report.forged_success_completions,
        Load64(&local_completion->slot.signal.sequence) == 0 ? 0 : 1
    );
    dsb(DSB_ALL);
}

#elif defined(PYPTO_FAULT_SERVICE)

FAULT_KERNEL_META(pypto_stage1c_fault_service_0_mix_aiv);

extern "C" __global__ __aicore__ void pypto_stage1c_fault_service_0_mix_aiv(
    __gm__ uint64_t *local_payload, __gm__ ControlBlock *local_submission,
    __gm__ uint64_t *remote_payload, __gm__ ControlBlock *remote_completion,
    uint64_t generation, uint64_t timeout_cycles, uint32_t payload_words, uint32_t mode
) {
    (void)local_payload;
    (void)remote_payload;
    (void)payload_words;
    const uint64_t wait_begin = static_cast<uint64_t>(get_sys_cnt());
    while (Load64(&local_submission->slot.signal.sequence) != 1 &&
           static_cast<uint64_t>(get_sys_cnt()) - wait_begin < timeout_cycles) {}
    dsb(DSB_ALL);
    Store64(&remote_completion->slot.message.status, kPausedStatus);
    dsb(DSB_ALL);
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    while (static_cast<uint64_t>(get_sys_cnt()) - begin < timeout_cycles) {}
    const uint64_t elapsed = static_cast<uint64_t>(get_sys_cnt()) - begin;
    PublishReport(&local_submission->report, elapsed, timeout_cycles, mode, 0);
    Store64(
        &local_submission->report.generation_errors,
        Load64(&local_submission->slot.message.generation) == generation ? 0 : 1
    );
    Store64(
        &local_submission->report.sequence_errors,
        Load64(&local_submission->slot.signal.sequence) == 1 ? 0 : 1
    );
    dsb(DSB_ALL);
}

#else
#error "Define PYPTO_FAULT_DRIVER or PYPTO_FAULT_SERVICE"
#endif
