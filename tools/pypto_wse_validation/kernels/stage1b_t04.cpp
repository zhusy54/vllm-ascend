// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// This file is a part of the vllm-ascend project.

// Pure CCEC AIV kernels for Stage 1B T04. Each endpoint launches exactly one
// kernel. The two kernels then execute all submission/completion traffic in
// Device Memory without per-sequence Host calls.
#include "cce_aicore_intrinsics.h"
#include <stdint.h>

namespace {

constexpr uint64_t kWaitTimeoutCycles = 20000000000ULL;
constexpr uint64_t kPatternMagic = 0x505950544f543034ULL;
constexpr uint64_t kGenerationMix = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t kSequenceMix = 0xd1b54a32d192ed03ULL;
constexpr uint64_t kIndexMix = 0x94d049bb133111ebULL;
constexpr uint64_t kByteRepeat = 0x0101010101010101ULL;

struct MetaHead {
    uint16_t type;
    uint16_t length;
};

struct KernelTypeMeta {
    MetaHead head;
    uint32_t kernel_type;
};

struct MixCoreMeta {
    MetaHead head;
    uint16_t aic_ratio;
    uint16_t aiv_ratio;
};

struct KernelMeta {
    KernelTypeMeta kernel_type;
    MixCoreMeta mix_core;
};

#define T04_KERNEL_META(kernel_name)                                                                        \
    static const KernelMeta g_##kernel_name##_meta __attribute__((used, section(".ascend.meta." #kernel_name))) = { \
        {{1, sizeof(uint32_t)}, 5}, {{3, sizeof(uint32_t)}, 0, 1}}

struct alignas(64) SignalLine {
    uint64_t sequence;
    uint64_t reserved[7];
};

struct alignas(64) MessageLine {
    uint64_t generation;
    uint64_t sequence;
    uint64_t payload_bytes;
    uint64_t checksum;
    uint64_t head_marker;
    uint64_t tail_marker;
    uint64_t status;
    uint64_t reserved;
};

struct alignas(64) ReportLine {
    uint64_t processed;
    uint64_t validation_errors;
    uint64_t sequence_errors;
    uint64_t generation_errors;
    uint64_t checksum_errors;
    uint64_t marker_errors;
    uint64_t timeouts;
    uint64_t elapsed_cycles;
};

struct alignas(64) ControlBlock {
    SignalLine signal;
    MessageLine message;
    ReportLine report;
};

static_assert(sizeof(SignalLine) == 64);
static_assert(sizeof(MessageLine) == 64);
static_assert(sizeof(ReportLine) == 64);
static_assert(sizeof(ControlBlock) == 192);

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

__aicore__ inline bool WaitForSequence(
    __gm__ uint64_t *signal, uint64_t expected, uint64_t &observed
) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    do {
        observed = Load64(signal);
        if (observed == expected) {
            return true;
        }
        if (observed > expected) {
            return false;
        }
    } while (static_cast<uint64_t>(get_sys_cnt()) - begin < kWaitTimeoutCycles);
    observed = Load64(signal);
    return observed == expected;
}

__aicore__ inline void PublishMessage(
    __gm__ ControlBlock *control, uint64_t generation, uint64_t sequence, uint64_t payload_bytes,
    uint64_t checksum, uint64_t head_marker, uint64_t tail_marker, uint64_t status
) {
    Store64(&control->message.generation, generation);
    Store64(&control->message.sequence, sequence);
    Store64(&control->message.payload_bytes, payload_bytes);
    Store64(&control->message.checksum, checksum);
    Store64(&control->message.head_marker, head_marker);
    Store64(&control->message.tail_marker, tail_marker);
    Store64(&control->message.status, status);
    dsb(DSB_ALL);
    Store64(&control->signal.sequence, sequence);
    dsb(DSB_ALL);
}

__aicore__ inline void PublishReport(
    __gm__ ReportLine *report, uint64_t processed, uint64_t validation_errors, uint64_t sequence_errors,
    uint64_t generation_errors, uint64_t checksum_errors, uint64_t marker_errors, uint64_t timeouts,
    uint64_t elapsed_cycles
) {
    Store64(&report->processed, processed);
    Store64(&report->validation_errors, validation_errors);
    Store64(&report->sequence_errors, sequence_errors);
    Store64(&report->generation_errors, generation_errors);
    Store64(&report->checksum_errors, checksum_errors);
    Store64(&report->marker_errors, marker_errors);
    Store64(&report->timeouts, timeouts);
    Store64(&report->elapsed_cycles, elapsed_cycles);
    dsb(DSB_ALL);
}

}  // namespace

#if defined(PYPTO_T04_DRIVER)

T04_KERNEL_META(pypto_stage1b_t04_driver_0_mix_aiv);

extern "C" __global__ __aicore__ void pypto_stage1b_t04_driver_0_mix_aiv(
    __gm__ uint64_t *local_output, __gm__ ControlBlock *local_completion,
    __gm__ uint64_t *remote_input, __gm__ ControlBlock *remote_submission,
    uint64_t generation, uint32_t payload_words, uint32_t sequence_count
) {
    uint64_t processed = 0;
    uint64_t validation_errors = 0;
    uint64_t sequence_errors = 0;
    uint64_t generation_errors = 0;
    uint64_t checksum_errors = 0;
    uint64_t marker_errors = 0;
    uint64_t timeouts = 0;
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());

    for (uint64_t sequence = 1; sequence <= sequence_count; ++sequence) {
        uint64_t input_checksum = 0;
        for (uint64_t index = 0; index < payload_words; ++index) {
            const uint64_t value = PatternWord(generation, sequence, index);
            Store64(&remote_input[index], value);
            input_checksum += value;
        }
        const uint64_t input_head = PatternWord(generation, sequence, 0);
        const uint64_t input_tail = PatternWord(generation, sequence, payload_words - 1);
        dsb(DSB_ALL);
        PublishMessage(
            remote_submission, generation, sequence, payload_words * sizeof(uint64_t), input_checksum,
            input_head, input_tail, 0
        );

        uint64_t observed_sequence = 0;
        if (!WaitForSequence(&local_completion->signal.sequence, sequence, observed_sequence)) {
            ++timeouts;
            if (observed_sequence > sequence) {
                ++sequence_errors;
            }
            break;
        }
        dsb(DSB_ALL);

        const uint64_t completion_generation = Load64(&local_completion->message.generation);
        const uint64_t completion_sequence = Load64(&local_completion->message.sequence);
        const uint64_t completion_bytes = Load64(&local_completion->message.payload_bytes);
        const uint64_t completion_checksum = Load64(&local_completion->message.checksum);
        const uint64_t completion_head = Load64(&local_completion->message.head_marker);
        const uint64_t completion_tail = Load64(&local_completion->message.tail_marker);
        const uint64_t completion_status = Load64(&local_completion->message.status);
        if (completion_generation != generation) {
            ++generation_errors;
        }
        if (completion_sequence != sequence) {
            ++sequence_errors;
        }
        if (completion_bytes != payload_words * sizeof(uint64_t) || completion_status != 0) {
            ++validation_errors;
        }

        const uint64_t mask = TransformMask(sequence);
        uint64_t observed_checksum = 0;
        uint64_t payload_mismatches = 0;
        for (uint64_t index = 0; index < payload_words; ++index) {
            const uint64_t observed = Load64(&local_output[index]);
            const uint64_t expected = PatternWord(generation, sequence, index) ^ mask;
            observed_checksum += observed;
            if (observed != expected) {
                ++payload_mismatches;
            }
        }
        if (payload_mismatches != 0 || completion_checksum != observed_checksum) {
            ++checksum_errors;
        }
        if (completion_head != (PatternWord(generation, sequence, 0) ^ mask) ||
            completion_tail != (PatternWord(generation, sequence, payload_words - 1) ^ mask)) {
            ++marker_errors;
        }
        validation_errors += payload_mismatches;
        ++processed;
    }

    validation_errors += sequence_errors + generation_errors + checksum_errors + marker_errors + timeouts;
    PublishReport(
        &local_completion->report, processed, validation_errors, sequence_errors, generation_errors,
        checksum_errors, marker_errors, timeouts, static_cast<uint64_t>(get_sys_cnt()) - begin
    );
}

#elif defined(PYPTO_T04_SERVICE)

T04_KERNEL_META(pypto_stage1b_t04_service_0_mix_aiv);

extern "C" __global__ __aicore__ void pypto_stage1b_t04_service_0_mix_aiv(
    __gm__ uint64_t *local_input, __gm__ ControlBlock *local_submission,
    __gm__ uint64_t *remote_output, __gm__ ControlBlock *remote_completion,
    uint64_t generation, uint32_t payload_words, uint32_t sequence_count
) {
    uint64_t processed = 0;
    uint64_t validation_errors = 0;
    uint64_t sequence_errors = 0;
    uint64_t generation_errors = 0;
    uint64_t checksum_errors = 0;
    uint64_t marker_errors = 0;
    uint64_t timeouts = 0;
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());

    for (uint64_t sequence = 1; sequence <= sequence_count; ++sequence) {
        uint64_t observed_sequence = 0;
        if (!WaitForSequence(&local_submission->signal.sequence, sequence, observed_sequence)) {
            ++timeouts;
            if (observed_sequence > sequence) {
                ++sequence_errors;
            }
            break;
        }
        dsb(DSB_ALL);

        const uint64_t submission_generation = Load64(&local_submission->message.generation);
        const uint64_t submission_sequence = Load64(&local_submission->message.sequence);
        const uint64_t submission_bytes = Load64(&local_submission->message.payload_bytes);
        const uint64_t submission_checksum = Load64(&local_submission->message.checksum);
        const uint64_t submission_head = Load64(&local_submission->message.head_marker);
        const uint64_t submission_tail = Load64(&local_submission->message.tail_marker);
        if (submission_generation != generation) {
            ++generation_errors;
        }
        if (submission_sequence != sequence) {
            ++sequence_errors;
        }
        if (submission_bytes != payload_words * sizeof(uint64_t)) {
            ++validation_errors;
        }

        const uint64_t mask = TransformMask(sequence);
        uint64_t input_checksum = 0;
        uint64_t output_checksum = 0;
        uint64_t input_head = 0;
        uint64_t input_tail = 0;
        uint64_t output_head = 0;
        uint64_t output_tail = 0;
        uint64_t payload_mismatches = 0;
        for (uint64_t index = 0; index < payload_words; ++index) {
            const uint64_t input = Load64(&local_input[index]);
            const uint64_t output = input ^ mask;
            if (input != PatternWord(generation, sequence, index)) {
                ++payload_mismatches;
            }
            input_checksum += input;
            output_checksum += output;
            Store64(&remote_output[index], output);
            if (index == 0) {
                input_head = input;
                output_head = output;
            }
            if (index + 1 == payload_words) {
                input_tail = input;
                output_tail = output;
            }
        }
        if (submission_checksum != input_checksum) {
            ++checksum_errors;
        }
        if (submission_head != input_head || submission_tail != input_tail) {
            ++marker_errors;
        }
        validation_errors += payload_mismatches;
        const uint64_t status = validation_errors + sequence_errors + generation_errors + checksum_errors + marker_errors;
        dsb(DSB_ALL);
        PublishMessage(
            remote_completion, generation, sequence, payload_words * sizeof(uint64_t), output_checksum,
            output_head, output_tail, status
        );
        ++processed;
    }

    validation_errors += sequence_errors + generation_errors + checksum_errors + marker_errors + timeouts;
    PublishReport(
        &local_submission->report, processed, validation_errors, sequence_errors, generation_errors,
        checksum_errors, marker_errors, timeouts, static_cast<uint64_t>(get_sys_cnt()) - begin
    );
}

#else
#error "Define PYPTO_T04_DRIVER or PYPTO_T04_SERVICE"
#endif
