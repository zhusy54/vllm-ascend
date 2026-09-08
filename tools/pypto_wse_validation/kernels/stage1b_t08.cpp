// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// This file is a part of the vllm-ascend project.

// Pure CCEC AIV kernels for Stage 1B T08. Mode 0 completes generation G.
// Mode 1 runs generation G+1 while injecting a descriptor and completion from
// G, proving that stale traffic neither consumes the current slot nor releases
// its sole credit.
#include "cce_aicore_intrinsics.h"
#include <stdint.h>

namespace {

constexpr uint64_t kWaitTimeoutCycles = 20000000000ULL;
constexpr uint64_t kPatternMagic = 0x505950544f543038ULL;
constexpr uint64_t kGenerationMix = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t kSequenceMix = 0xd1b54a32d192ed03ULL;
constexpr uint64_t kIndexMix = 0x94d049bb133111ebULL;
constexpr uint64_t kByteRepeat = 0x0101010101010101ULL;
constexpr uint64_t kStaleSequence = 0x8000000000000001ULL;
constexpr uint64_t kStaleStatus = 2;

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

#define T08_KERNEL_META(kernel_name)                                                                        \
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
    uint64_t slot;
};

struct alignas(64) SlotControl {
    SignalLine signal;
    MessageLine message;
};

struct alignas(64) AckLine {
    uint64_t sequence;
    uint64_t reserved[7];
};

struct alignas(64) ReportBlock {
    uint64_t processed;
    uint64_t validation_errors;
    uint64_t sequence_errors;
    uint64_t generation_errors;
    uint64_t checksum_errors;
    uint64_t marker_errors;
    uint64_t timeouts;
    uint64_t elapsed_cycles;
    uint64_t submissions;
    uint64_t completions;
    uint64_t stale_descriptor_injections;
    uint64_t stale_descriptor_rejections;
    uint64_t stale_completion_injections;
    uint64_t stale_completion_rejections;
    uint64_t old_completion_credit_releases;
    uint64_t current_slot_preserved;
    uint64_t credits_acquired;
    uint64_t credits_returned;
    uint64_t terminal_tasks;
    uint64_t progress_after_stale;
    uint64_t input_fences;
    uint64_t output_fences;
    uint64_t mode;
    uint64_t reserved;
};

struct alignas(64) ControlBlock {
    SlotControl normal;
    SlotControl stale;
    AckLine ack;
    ReportBlock report;
};

struct Counters {
    uint64_t processed;
    uint64_t validation_errors;
    uint64_t sequence_errors;
    uint64_t generation_errors;
    uint64_t checksum_errors;
    uint64_t marker_errors;
    uint64_t timeouts;
    uint64_t submissions;
    uint64_t completions;
    uint64_t stale_descriptor_injections;
    uint64_t stale_descriptor_rejections;
    uint64_t stale_completion_injections;
    uint64_t stale_completion_rejections;
    uint64_t old_completion_credit_releases;
    uint64_t current_slot_preserved;
    uint64_t credits_acquired;
    uint64_t credits_returned;
    uint64_t terminal_tasks;
    uint64_t progress_after_stale;
    uint64_t input_fences;
    uint64_t output_fences;
};

static_assert(sizeof(SignalLine) == 64);
static_assert(sizeof(MessageLine) == 64);
static_assert(sizeof(SlotControl) == 128);
static_assert(sizeof(AckLine) == 64);
static_assert(sizeof(ReportBlock) == 192);
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

__aicore__ inline bool WaitForExact(__gm__ uint64_t *signal, uint64_t expected, uint64_t &observed) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    do {
        observed = Load64(signal);
        if (observed == expected) {
            return true;
        }
    } while (static_cast<uint64_t>(get_sys_cnt()) - begin < kWaitTimeoutCycles);
    observed = Load64(signal);
    return observed == expected;
}

__aicore__ inline void PublishMessage(
    __gm__ SlotControl *control, uint64_t generation, uint64_t sequence, uint64_t payload_bytes,
    uint64_t checksum, uint64_t head_marker, uint64_t tail_marker, uint64_t status
) {
    Store64(&control->message.generation, generation);
    Store64(&control->message.sequence, sequence);
    Store64(&control->message.payload_bytes, payload_bytes);
    Store64(&control->message.checksum, checksum);
    Store64(&control->message.head_marker, head_marker);
    Store64(&control->message.tail_marker, tail_marker);
    Store64(&control->message.status, status);
    Store64(&control->message.slot, 0);
    dsb(DSB_ALL);
    Store64(&control->signal.sequence, sequence);
    dsb(DSB_ALL);
}

__aicore__ inline void PublishReport(
    __gm__ ReportBlock *report, const Counters &counters, uint64_t elapsed_cycles, uint64_t mode
) {
    Store64(&report->processed, counters.processed);
    Store64(
        &report->validation_errors,
        counters.validation_errors + counters.sequence_errors + counters.generation_errors +
            counters.checksum_errors + counters.marker_errors + counters.timeouts +
            counters.old_completion_credit_releases
    );
    Store64(&report->sequence_errors, counters.sequence_errors);
    Store64(&report->generation_errors, counters.generation_errors);
    Store64(&report->checksum_errors, counters.checksum_errors);
    Store64(&report->marker_errors, counters.marker_errors);
    Store64(&report->timeouts, counters.timeouts);
    Store64(&report->elapsed_cycles, elapsed_cycles);
    Store64(&report->submissions, counters.submissions);
    Store64(&report->completions, counters.completions);
    Store64(&report->stale_descriptor_injections, counters.stale_descriptor_injections);
    Store64(&report->stale_descriptor_rejections, counters.stale_descriptor_rejections);
    Store64(&report->stale_completion_injections, counters.stale_completion_injections);
    Store64(&report->stale_completion_rejections, counters.stale_completion_rejections);
    Store64(&report->old_completion_credit_releases, counters.old_completion_credit_releases);
    Store64(&report->current_slot_preserved, counters.current_slot_preserved);
    Store64(&report->credits_acquired, counters.credits_acquired);
    Store64(&report->credits_returned, counters.credits_returned);
    Store64(&report->terminal_tasks, counters.terminal_tasks);
    Store64(&report->progress_after_stale, counters.progress_after_stale);
    Store64(&report->input_fences, counters.input_fences);
    Store64(&report->output_fences, counters.output_fences);
    Store64(&report->mode, mode);
    Store64(&report->reserved, 0);
    dsb(DSB_ALL);
}

}  // namespace

#if defined(PYPTO_T08_DRIVER)

T08_KERNEL_META(pypto_stage1b_t08_driver_0_mix_aiv);

namespace {

__aicore__ inline void SubmitCurrent(
    __gm__ uint64_t *remote_input, __gm__ SlotControl *remote_submission, uint64_t generation,
    uint64_t sequence, uint64_t payload_words, Counters &counters
) {
    uint64_t checksum = 0;
    for (uint64_t index = 0; index < payload_words; ++index) {
        const uint64_t value = PatternWord(generation, sequence, index);
        Store64(&remote_input[index], value);
        checksum += value;
    }
    dsb(DSB_ALL);
    PublishMessage(
        remote_submission, generation, sequence, payload_words * sizeof(uint64_t), checksum,
        PatternWord(generation, sequence, 0), PatternWord(generation, sequence, payload_words - 1), 0
    );
    ++counters.submissions;
    ++counters.credits_acquired;
    ++counters.input_fences;
}

__aicore__ inline bool CompleteCurrent(
    __gm__ uint64_t *local_output, __gm__ SlotControl *local_completion, uint64_t generation,
    uint64_t sequence, uint64_t payload_words, Counters &counters
) {
    uint64_t observed_sequence = 0;
    if (!WaitForExact(&local_completion->signal.sequence, sequence, observed_sequence)) {
        ++counters.timeouts;
        return false;
    }
    dsb(DSB_ALL);
    __gm__ MessageLine *message = &local_completion->message;
    const uint64_t completion_generation = Load64(&message->generation);
    const uint64_t completion_sequence = Load64(&message->sequence);
    const uint64_t completion_bytes = Load64(&message->payload_bytes);
    const uint64_t completion_checksum = Load64(&message->checksum);
    const uint64_t completion_head = Load64(&message->head_marker);
    const uint64_t completion_tail = Load64(&message->tail_marker);
    const uint64_t completion_status = Load64(&message->status);
    if (completion_generation != generation) {
        ++counters.generation_errors;
    }
    if (completion_sequence != sequence) {
        ++counters.sequence_errors;
    }
    if (completion_bytes != payload_words * sizeof(uint64_t) || completion_status != 0) {
        ++counters.validation_errors;
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
        ++counters.checksum_errors;
    }
    if (completion_head != (PatternWord(generation, sequence, 0) ^ mask) ||
        completion_tail != (PatternWord(generation, sequence, payload_words - 1) ^ mask)) {
        ++counters.marker_errors;
    }
    counters.validation_errors += payload_mismatches;
    ++counters.completions;
    ++counters.processed;
    ++counters.credits_returned;
    ++counters.terminal_tasks;
    return true;
}

}  // namespace

extern "C" __global__ __aicore__ void pypto_stage1b_t08_driver_0_mix_aiv(
    __gm__ uint64_t *local_output, __gm__ ControlBlock *local_completion,
    __gm__ uint64_t *remote_input, __gm__ ControlBlock *remote_submission,
    uint64_t generation, uint64_t previous_generation, uint32_t payload_words,
    uint32_t sequence_count, uint32_t mode
) {
    Counters counters = {};
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());

    if (mode == 0) {
        for (uint64_t sequence = 1; sequence <= sequence_count; ++sequence) {
            SubmitCurrent(remote_input, &remote_submission->normal, generation, sequence, payload_words, counters);
            if (!CompleteCurrent(local_output, &local_completion->normal, generation, sequence, payload_words, counters)) {
                break;
            }
        }
    } else if (mode == 1 && sequence_count == 1 && previous_generation + 1 == generation) {
        SubmitCurrent(remote_input, &remote_submission->normal, generation, 1, payload_words, counters);
        PublishMessage(
            &remote_submission->stale, previous_generation, kStaleSequence, 0, 0, 0, 0, 0
        );
        ++counters.stale_descriptor_injections;

        uint64_t observed_stale = 0;
        if (!WaitForExact(&local_completion->stale.signal.sequence, kStaleSequence, observed_stale)) {
            ++counters.timeouts;
        } else {
            dsb(DSB_ALL);
            __gm__ MessageLine *stale = &local_completion->stale.message;
            if (Load64(&stale->generation) != previous_generation) {
                ++counters.generation_errors;
            }
            if (Load64(&stale->sequence) != kStaleSequence) {
                ++counters.sequence_errors;
            }
            if (Load64(&stale->status) != kStaleStatus) {
                ++counters.validation_errors;
            } else {
                ++counters.stale_completion_rejections;
            }

            if (Load64(&local_completion->normal.signal.sequence) != 0) {
                ++counters.old_completion_credit_releases;
            } else {
                counters.current_slot_preserved = 1;
            }
            Store64(&remote_submission->ack.sequence, kStaleSequence);
            dsb(DSB_ALL);
            if (CompleteCurrent(local_output, &local_completion->normal, generation, 1, payload_words, counters)) {
                counters.progress_after_stale = 1;
            }
        }
    } else {
        ++counters.validation_errors;
    }

    PublishReport(
        &local_completion->report, counters, static_cast<uint64_t>(get_sys_cnt()) - begin, mode
    );
}

#elif defined(PYPTO_T08_SERVICE)

T08_KERNEL_META(pypto_stage1b_t08_service_0_mix_aiv);

namespace {

__aicore__ inline bool WaitForCurrent(
    __gm__ SlotControl *local_submission, uint64_t sequence, Counters &counters
) {
    uint64_t observed_sequence = 0;
    if (!WaitForExact(&local_submission->signal.sequence, sequence, observed_sequence)) {
        ++counters.timeouts;
        return false;
    }
    dsb(DSB_ALL);
    ++counters.submissions;
    return true;
}

__aicore__ inline void ProcessCurrent(
    __gm__ uint64_t *local_input, __gm__ SlotControl *local_submission,
    __gm__ uint64_t *remote_output, __gm__ SlotControl *remote_completion,
    uint64_t generation, uint64_t sequence, uint64_t payload_words, Counters &counters
) {
    __gm__ MessageLine *message = &local_submission->message;
    const uint64_t submission_generation = Load64(&message->generation);
    const uint64_t submission_sequence = Load64(&message->sequence);
    const uint64_t submission_bytes = Load64(&message->payload_bytes);
    const uint64_t submission_checksum = Load64(&message->checksum);
    const uint64_t submission_head = Load64(&message->head_marker);
    const uint64_t submission_tail = Load64(&message->tail_marker);
    if (submission_generation != generation) {
        ++counters.generation_errors;
    }
    if (submission_sequence != sequence) {
        ++counters.sequence_errors;
    }
    if (submission_bytes != payload_words * sizeof(uint64_t)) {
        ++counters.validation_errors;
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
        ++counters.checksum_errors;
    }
    if (submission_head != input_head || submission_tail != input_tail) {
        ++counters.marker_errors;
    }
    counters.validation_errors += payload_mismatches;
    const uint64_t status = counters.validation_errors + counters.sequence_errors +
        counters.generation_errors + counters.checksum_errors + counters.marker_errors;
    dsb(DSB_ALL);
    PublishMessage(
        remote_completion, generation, sequence, payload_words * sizeof(uint64_t), output_checksum,
        output_head, output_tail, status
    );
    ++counters.completions;
    ++counters.processed;
    ++counters.terminal_tasks;
    ++counters.output_fences;
}

}  // namespace

extern "C" __global__ __aicore__ void pypto_stage1b_t08_service_0_mix_aiv(
    __gm__ uint64_t *local_input, __gm__ ControlBlock *local_submission,
    __gm__ uint64_t *remote_output, __gm__ ControlBlock *remote_completion,
    uint64_t generation, uint64_t previous_generation, uint32_t payload_words,
    uint32_t sequence_count, uint32_t mode
) {
    Counters counters = {};
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());

    if (mode == 0) {
        for (uint64_t sequence = 1; sequence <= sequence_count; ++sequence) {
            if (!WaitForCurrent(&local_submission->normal, sequence, counters)) {
                break;
            }
            ProcessCurrent(
                local_input, &local_submission->normal, remote_output, &remote_completion->normal,
                generation, sequence, payload_words, counters
            );
        }
    } else if (mode == 1 && sequence_count == 1 && previous_generation + 1 == generation) {
        if (WaitForCurrent(&local_submission->normal, 1, counters)) {
            uint64_t observed_stale = 0;
            if (!WaitForExact(&local_submission->stale.signal.sequence, kStaleSequence, observed_stale)) {
                ++counters.timeouts;
            } else {
                dsb(DSB_ALL);
                __gm__ MessageLine *stale = &local_submission->stale.message;
                if (Load64(&stale->generation) == previous_generation &&
                    Load64(&stale->generation) != generation) {
                    ++counters.stale_descriptor_rejections;
                } else {
                    ++counters.generation_errors;
                }
                if (Load64(&stale->sequence) != kStaleSequence) {
                    ++counters.sequence_errors;
                }
                if (Load64(&stale->status) != 0) {
                    ++counters.validation_errors;
                }

                if (Load64(&local_submission->normal.signal.sequence) == 1 &&
                    Load64(&local_submission->normal.message.generation) == generation &&
                    Load64(&local_submission->normal.message.sequence) == 1) {
                    counters.current_slot_preserved = 1;
                } else {
                    ++counters.validation_errors;
                }
                PublishMessage(
                    &remote_completion->stale, previous_generation, kStaleSequence, 0, 0, 0, 0,
                    kStaleStatus
                );
                ++counters.stale_completion_injections;

                uint64_t observed_ack = 0;
                if (!WaitForExact(&local_submission->ack.sequence, kStaleSequence, observed_ack)) {
                    ++counters.timeouts;
                } else {
                    ProcessCurrent(
                        local_input, &local_submission->normal, remote_output,
                        &remote_completion->normal, generation, 1, payload_words, counters
                    );
                    counters.progress_after_stale = 1;
                }
            }
        }
    } else {
        ++counters.validation_errors;
    }

    PublishReport(
        &local_submission->report, counters, static_cast<uint64_t>(get_sys_cnt()) - begin, mode
    );
}

#else
#error "Define PYPTO_T08_DRIVER or PYPTO_T08_SERVICE"
#endif
