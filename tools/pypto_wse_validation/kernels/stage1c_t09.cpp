// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// This file is a part of the vllm-ascend project.

// Stage 1C T09: 100 warmups followed by 10,000 device-driven round trips.
#include "cce_aicore_intrinsics.h"
#include <stdint.h>

namespace {

constexpr uint64_t kWaitTimeoutCycles = 30000000000ULL;
constexpr uint64_t kPatternMagic = 0x505950544f433039ULL;
constexpr uint64_t kGenerationMix = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t kSequenceMix = 0xd1b54a32d192ed03ULL;
constexpr uint64_t kIndexMix = 0x94d049bb133111ebULL;
constexpr uint64_t kByteRepeat = 0x0101010101010101ULL;
constexpr uint64_t kSlotStrideWords = (1024ULL * 1024ULL) / sizeof(uint64_t);
constexpr uint64_t kWarmupCount = 100;
constexpr uint64_t kProgressInterval = 100;

struct MetaHead { uint16_t type; uint16_t length; };
struct KernelTypeMeta { MetaHead head; uint32_t kernel_type; };
struct MixCoreMeta { MetaHead head; uint16_t aic_ratio; uint16_t aiv_ratio; };
struct KernelMeta { KernelTypeMeta kernel_type; MixCoreMeta mix_core; };

#define T09_KERNEL_META(kernel_name)                                                                        \
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
    uint64_t processed;
    uint64_t warmup_processed;
    uint64_t measured_processed;
    uint64_t validation_errors;
    uint64_t sequence_errors;
    uint64_t generation_errors;
    uint64_t checksum_errors;
    uint64_t marker_errors;
    uint64_t timeouts;
    uint64_t elapsed_cycles;
    uint64_t slot0_processed;
    uint64_t slot1_processed;
    uint64_t submissions;
    uint64_t completions;
    uint64_t credits_acquired;
    uint64_t credits_returned;
    uint64_t terminal_tasks;
    uint64_t max_inflight;
    uint64_t queue_stalls;
    uint64_t progress_checkpoints;
    uint64_t progress_interval;
    uint64_t payload_64_count;
    uint64_t payload_4k_count;
    uint64_t payload_64k_count;
    uint64_t payload_1m_count;
    uint64_t payload_words_validated;
    uint64_t input_fences;
    uint64_t output_fences;
    uint64_t reserved[4];
};
struct alignas(64) ControlBlock { SlotControl slots[2]; ReportBlock report; };
struct Counters {
    uint64_t processed;
    uint64_t warmup_processed;
    uint64_t measured_processed;
    uint64_t validation_errors;
    uint64_t sequence_errors;
    uint64_t generation_errors;
    uint64_t checksum_errors;
    uint64_t marker_errors;
    uint64_t timeouts;
    uint64_t slot_processed[2];
    uint64_t submissions;
    uint64_t completions;
    uint64_t credits_acquired;
    uint64_t credits_returned;
    uint64_t terminal_tasks;
    uint64_t max_inflight;
    uint64_t queue_stalls;
    uint64_t progress_checkpoints;
    uint64_t payload_count[4];
    uint64_t payload_words_validated;
    uint64_t input_fences;
    uint64_t output_fences;
};

static_assert(sizeof(SignalLine) == 64);
static_assert(sizeof(MessageLine) == 64);
static_assert(sizeof(SlotControl) == 128);
static_assert(sizeof(ReportBlock) == 256);
static_assert(sizeof(ControlBlock) == 512);

__aicore__ inline void Store64(__gm__ uint64_t *address, uint64_t value) {
    __builtin_cce_st_dev(value, address, 0);
}
__aicore__ inline uint64_t Load64(__gm__ uint64_t *address) {
    return static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
}
__aicore__ inline uint64_t PatternWord(uint64_t generation, uint64_t sequence, uint64_t index) {
    return kPatternMagic ^ (generation * kGenerationMix) ^ (sequence * kSequenceMix) ^ (index * kIndexMix);
}
__aicore__ inline uint64_t TransformMask(uint64_t sequence) {
    return (sequence & 0xffULL) * kByteRepeat;
}
__aicore__ inline __gm__ uint64_t *PayloadSlot(__gm__ uint64_t *payload, uint64_t slot) {
    return payload + slot * kSlotStrideWords;
}
__aicore__ inline uint64_t PayloadWords(uint64_t sequence, uint64_t &payload_index) {
    if (sequence <= kWarmupCount) {
        payload_index = 1;
        return (4ULL * 1024ULL) / sizeof(uint64_t);
    }
    const uint64_t measured_sequence = sequence - kWarmupCount;
    if (measured_sequence % 100 == 0) {
        payload_index = 3;
        return (1024ULL * 1024ULL) / sizeof(uint64_t);
    }
    if (measured_sequence % 5 == 0) {
        payload_index = 2;
        return (64ULL * 1024ULL) / sizeof(uint64_t);
    }
    if (measured_sequence % 2 == 0) {
        payload_index = 1;
        return (4ULL * 1024ULL) / sizeof(uint64_t);
    }
    payload_index = 0;
    return 64ULL / sizeof(uint64_t);
}
__aicore__ inline bool WaitForSequence(__gm__ uint64_t *signal, uint64_t expected) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    do {
        if (Load64(signal) == expected) return true;
    } while (static_cast<uint64_t>(get_sys_cnt()) - begin < kWaitTimeoutCycles);
    return Load64(signal) == expected;
}
__aicore__ inline void PublishMessage(
    __gm__ SlotControl *control, uint64_t generation, uint64_t sequence, uint64_t payload_bytes,
    uint64_t checksum, uint64_t head_marker, uint64_t tail_marker, uint64_t status, uint64_t slot
) {
    Store64(&control->message.generation, generation);
    Store64(&control->message.sequence, sequence);
    Store64(&control->message.payload_bytes, payload_bytes);
    Store64(&control->message.checksum, checksum);
    Store64(&control->message.head_marker, head_marker);
    Store64(&control->message.tail_marker, tail_marker);
    Store64(&control->message.status, status);
    Store64(&control->message.slot, slot);
    dsb(DSB_ALL);
    Store64(&control->signal.sequence, sequence);
    dsb(DSB_ALL);
}
__aicore__ inline void AccountProgress(Counters &counters, uint64_t sequence, uint64_t payload_index) {
    ++counters.processed;
    ++counters.terminal_tasks;
    if (sequence <= kWarmupCount) {
        ++counters.warmup_processed;
    } else {
        ++counters.measured_processed;
        ++counters.payload_count[payload_index];
        if (counters.measured_processed % kProgressInterval == 0) ++counters.progress_checkpoints;
    }
}
__aicore__ inline void PublishReport(__gm__ ReportBlock *report, const Counters &c, uint64_t elapsed) {
    Store64(&report->processed, c.processed);
    Store64(&report->warmup_processed, c.warmup_processed);
    Store64(&report->measured_processed, c.measured_processed);
    Store64(&report->validation_errors, c.validation_errors + c.sequence_errors + c.generation_errors +
        c.checksum_errors + c.marker_errors + c.timeouts);
    Store64(&report->sequence_errors, c.sequence_errors);
    Store64(&report->generation_errors, c.generation_errors);
    Store64(&report->checksum_errors, c.checksum_errors);
    Store64(&report->marker_errors, c.marker_errors);
    Store64(&report->timeouts, c.timeouts);
    Store64(&report->elapsed_cycles, elapsed);
    Store64(&report->slot0_processed, c.slot_processed[0]);
    Store64(&report->slot1_processed, c.slot_processed[1]);
    Store64(&report->submissions, c.submissions);
    Store64(&report->completions, c.completions);
    Store64(&report->credits_acquired, c.credits_acquired);
    Store64(&report->credits_returned, c.credits_returned);
    Store64(&report->terminal_tasks, c.terminal_tasks);
    Store64(&report->max_inflight, c.max_inflight);
    Store64(&report->queue_stalls, c.queue_stalls);
    Store64(&report->progress_checkpoints, c.progress_checkpoints);
    Store64(&report->progress_interval, kProgressInterval);
    Store64(&report->payload_64_count, c.payload_count[0]);
    Store64(&report->payload_4k_count, c.payload_count[1]);
    Store64(&report->payload_64k_count, c.payload_count[2]);
    Store64(&report->payload_1m_count, c.payload_count[3]);
    Store64(&report->payload_words_validated, c.payload_words_validated);
    Store64(&report->input_fences, c.input_fences);
    Store64(&report->output_fences, c.output_fences);
    for (uint64_t index = 0; index < 4; ++index) Store64(&report->reserved[index], 0);
    dsb(DSB_ALL);
}

}  // namespace

#if defined(PYPTO_T09_DRIVER)

T09_KERNEL_META(pypto_stage1c_t09_driver_0_mix_aiv);

namespace {
__aicore__ inline void Submit(
    __gm__ uint64_t *remote_payload, __gm__ SlotControl *remote_control,
    uint64_t generation, uint64_t sequence, uint64_t slot, Counters &counters
) {
    uint64_t payload_index = 0;
    const uint64_t words = PayloadWords(sequence, payload_index);
    __gm__ uint64_t *destination = PayloadSlot(remote_payload, slot);
    uint64_t checksum = 0;
    for (uint64_t index = 0; index < words; ++index) {
        const uint64_t value = PatternWord(generation, sequence, index);
        Store64(&destination[index], value);
        checksum += value;
    }
    dsb(DSB_ALL);
    PublishMessage(
        remote_control, generation, sequence, words * sizeof(uint64_t), checksum,
        PatternWord(generation, sequence, 0), PatternWord(generation, sequence, words - 1), 0, slot
    );
    ++counters.submissions;
    ++counters.credits_acquired;
    ++counters.input_fences;
}
__aicore__ inline bool Complete(
    __gm__ uint64_t *local_payload, __gm__ SlotControl *local_control,
    uint64_t generation, uint64_t sequence, uint64_t slot, Counters &counters
) {
    if (!WaitForSequence(&local_control->signal.sequence, sequence)) {
        ++counters.timeouts;
        return false;
    }
    dsb(DSB_ALL);
    uint64_t payload_index = 0;
    const uint64_t words = PayloadWords(sequence, payload_index);
    __gm__ MessageLine *message = &local_control->message;
    if (Load64(&message->generation) != generation) ++counters.generation_errors;
    if (Load64(&message->sequence) != sequence || Load64(&message->slot) != slot) ++counters.sequence_errors;
    if (Load64(&message->payload_bytes) != words * sizeof(uint64_t) || Load64(&message->status) != 0)
        ++counters.validation_errors;
    const uint64_t mask = TransformMask(sequence);
    __gm__ uint64_t *source = PayloadSlot(local_payload, slot);
    uint64_t checksum = 0;
    uint64_t mismatches = 0;
    for (uint64_t index = 0; index < words; ++index) {
        const uint64_t value = Load64(&source[index]);
        checksum += value;
        if (value != (PatternWord(generation, sequence, index) ^ mask)) ++mismatches;
    }
    if (mismatches != 0 || checksum != Load64(&message->checksum)) ++counters.checksum_errors;
    if (Load64(&message->head_marker) != (PatternWord(generation, sequence, 0) ^ mask) ||
        Load64(&message->tail_marker) != (PatternWord(generation, sequence, words - 1) ^ mask))
        ++counters.marker_errors;
    counters.validation_errors += mismatches;
    counters.payload_words_validated += words;
    ++counters.completions;
    ++counters.credits_returned;
    ++counters.slot_processed[slot];
    AccountProgress(counters, sequence, payload_index);
    return true;
}
}  // namespace

extern "C" __global__ __aicore__ void pypto_stage1c_t09_driver_0_mix_aiv(
    __gm__ uint64_t *local_payload, __gm__ ControlBlock *local_completion,
    __gm__ uint64_t *remote_payload, __gm__ ControlBlock *remote_submission,
    uint64_t generation, uint32_t total_count
) {
    Counters counters = {};
    counters.max_inflight = 2;
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    for (uint64_t first = 1; first <= total_count; first += 2) {
        const uint64_t second = first + 1;
        Submit(remote_payload, &remote_submission->slots[0], generation, first, 0, counters);
        Submit(remote_payload, &remote_submission->slots[1], generation, second, 1, counters);
        if (!Complete(local_payload, &local_completion->slots[1], generation, second, 1, counters)) break;
        if (!Complete(local_payload, &local_completion->slots[0], generation, first, 0, counters)) break;
    }
    PublishReport(&local_completion->report, counters, static_cast<uint64_t>(get_sys_cnt()) - begin);
}

#elif defined(PYPTO_T09_SERVICE)

T09_KERNEL_META(pypto_stage1c_t09_service_0_mix_aiv);

namespace {
__aicore__ inline bool Process(
    __gm__ uint64_t *local_payload, __gm__ SlotControl *local_control,
    __gm__ uint64_t *remote_payload, __gm__ SlotControl *remote_control,
    uint64_t generation, uint64_t sequence, uint64_t slot, Counters &counters
) {
    if (!WaitForSequence(&local_control->signal.sequence, sequence)) {
        ++counters.timeouts;
        return false;
    }
    dsb(DSB_ALL);
    uint64_t payload_index = 0;
    const uint64_t words = PayloadWords(sequence, payload_index);
    __gm__ MessageLine *message = &local_control->message;
    if (Load64(&message->generation) != generation) ++counters.generation_errors;
    if (Load64(&message->sequence) != sequence || Load64(&message->slot) != slot) ++counters.sequence_errors;
    if (Load64(&message->payload_bytes) != words * sizeof(uint64_t)) ++counters.validation_errors;
    __gm__ uint64_t *source = PayloadSlot(local_payload, slot);
    __gm__ uint64_t *destination = PayloadSlot(remote_payload, slot);
    const uint64_t mask = TransformMask(sequence);
    uint64_t input_checksum = 0;
    uint64_t output_checksum = 0;
    uint64_t mismatches = 0;
    for (uint64_t index = 0; index < words; ++index) {
        const uint64_t input = Load64(&source[index]);
        const uint64_t output = input ^ mask;
        if (input != PatternWord(generation, sequence, index)) ++mismatches;
        input_checksum += input;
        output_checksum += output;
        Store64(&destination[index], output);
    }
    if (mismatches != 0 || input_checksum != Load64(&message->checksum)) ++counters.checksum_errors;
    if (Load64(&message->head_marker) != PatternWord(generation, sequence, 0) ||
        Load64(&message->tail_marker) != PatternWord(generation, sequence, words - 1))
        ++counters.marker_errors;
    counters.validation_errors += mismatches;
    counters.payload_words_validated += words;
    dsb(DSB_ALL);
    PublishMessage(
        remote_control, generation, sequence, words * sizeof(uint64_t), output_checksum,
        PatternWord(generation, sequence, 0) ^ mask, PatternWord(generation, sequence, words - 1) ^ mask,
        counters.validation_errors + counters.sequence_errors + counters.generation_errors +
            counters.checksum_errors + counters.marker_errors,
        slot
    );
    ++counters.submissions;
    ++counters.completions;
    ++counters.output_fences;
    ++counters.slot_processed[slot];
    AccountProgress(counters, sequence, payload_index);
    return true;
}
}  // namespace

extern "C" __global__ __aicore__ void pypto_stage1c_t09_service_0_mix_aiv(
    __gm__ uint64_t *local_payload, __gm__ ControlBlock *local_submission,
    __gm__ uint64_t *remote_payload, __gm__ ControlBlock *remote_completion,
    uint64_t generation, uint32_t total_count
) {
    Counters counters = {};
    counters.max_inflight = 2;
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    for (uint64_t first = 1; first <= total_count; first += 2) {
        const uint64_t second = first + 1;
        if (!Process(local_payload, &local_submission->slots[1], remote_payload,
                     &remote_completion->slots[1], generation, second, 1, counters)) break;
        if (!Process(local_payload, &local_submission->slots[0], remote_payload,
                     &remote_completion->slots[0], generation, first, 0, counters)) break;
    }
    PublishReport(&local_submission->report, counters, static_cast<uint64_t>(get_sys_cnt()) - begin);
}

#else
#error "Define PYPTO_T09_DRIVER or PYPTO_T09_SERVICE"
#endif
