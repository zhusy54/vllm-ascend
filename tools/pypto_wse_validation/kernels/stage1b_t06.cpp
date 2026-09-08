// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// This file is a part of the vllm-ascend project.

// Pure CCEC AIV kernels for Stage 1B T06. The service waits for a device-side
// backpressure probe before consuming the first two submissions. The driver
// rejects sequence 3 once with NO_CREDIT, retains it as pending, and does not
// retry submission until a real completion returns a credit.
#include "cce_aicore_intrinsics.h"
#include <stdint.h>

namespace {

constexpr uint64_t kWaitTimeoutCycles = 20000000000ULL;
constexpr uint64_t kPatternMagic = 0x505950544f543036ULL;
constexpr uint64_t kGenerationMix = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t kSequenceMix = 0xd1b54a32d192ed03ULL;
constexpr uint64_t kIndexMix = 0x94d049bb133111ebULL;
constexpr uint64_t kByteRepeat = 0x0101010101010101ULL;
constexpr uint64_t kSlotStrideWords = (1024ULL * 1024ULL) / sizeof(uint64_t);
constexpr uint64_t kNoCreditStatus = 1;

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

#define T06_KERNEL_META(kernel_name)                                                                        \
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

struct alignas(64) BackpressureLine {
    uint64_t pending_sequence;
    uint64_t queue_depth;
    uint64_t status;
    uint64_t submission_retry_spins;
    uint64_t service_resume_sequence;
    uint64_t reserved[3];
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
    uint64_t slot0_processed;
    uint64_t slot1_processed;
    uint64_t credits_acquired;
    uint64_t terminal_tasks;
    uint64_t credits_returned;
    uint64_t max_inflight;
    uint64_t slot_overwrite_errors;
    uint64_t no_credit_events;
    uint64_t pending_requests;
    uint64_t submission_retry_spins;
    uint64_t service_pause_observed;
    uint64_t service_resume_observed;
    uint64_t progress_after_resume;
    uint64_t peak_queue_depth;
    uint64_t submissions;
    uint64_t completions;
};

struct alignas(64) ControlBlock {
    SlotControl slots[2];
    BackpressureLine backpressure;
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
    uint64_t slot_processed[2];
    uint64_t credits_acquired;
    uint64_t terminal_tasks;
    uint64_t credits_returned;
    uint64_t max_inflight;
    uint64_t slot_overwrite_errors;
    uint64_t no_credit_events;
    uint64_t pending_requests;
    uint64_t submission_retry_spins;
    uint64_t service_pause_observed;
    uint64_t service_resume_observed;
    uint64_t progress_after_resume;
    uint64_t peak_queue_depth;
    uint64_t submissions;
    uint64_t completions;
};

static_assert(sizeof(SignalLine) == 64);
static_assert(sizeof(MessageLine) == 64);
static_assert(sizeof(SlotControl) == 128);
static_assert(sizeof(BackpressureLine) == 64);
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

__aicore__ inline __gm__ uint64_t *PayloadSlot(__gm__ uint64_t *payload, uint64_t slot) {
    return payload + slot * kSlotStrideWords;
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

__aicore__ inline void PublishReport(__gm__ ReportBlock *report, const Counters &counters, uint64_t elapsed_cycles) {
    Store64(&report->processed, counters.processed);
    Store64(&report->validation_errors, counters.validation_errors + counters.sequence_errors +
        counters.generation_errors + counters.checksum_errors + counters.marker_errors + counters.timeouts +
        counters.slot_overwrite_errors);
    Store64(&report->sequence_errors, counters.sequence_errors);
    Store64(&report->generation_errors, counters.generation_errors);
    Store64(&report->checksum_errors, counters.checksum_errors);
    Store64(&report->marker_errors, counters.marker_errors);
    Store64(&report->timeouts, counters.timeouts);
    Store64(&report->elapsed_cycles, elapsed_cycles);
    Store64(&report->slot0_processed, counters.slot_processed[0]);
    Store64(&report->slot1_processed, counters.slot_processed[1]);
    Store64(&report->credits_acquired, counters.credits_acquired);
    Store64(&report->terminal_tasks, counters.terminal_tasks);
    Store64(&report->credits_returned, counters.credits_returned);
    Store64(&report->max_inflight, counters.max_inflight);
    Store64(&report->slot_overwrite_errors, counters.slot_overwrite_errors);
    Store64(&report->no_credit_events, counters.no_credit_events);
    Store64(&report->pending_requests, counters.pending_requests);
    Store64(&report->submission_retry_spins, counters.submission_retry_spins);
    Store64(&report->service_pause_observed, counters.service_pause_observed);
    Store64(&report->service_resume_observed, counters.service_resume_observed);
    Store64(&report->progress_after_resume, counters.progress_after_resume);
    Store64(&report->peak_queue_depth, counters.peak_queue_depth);
    Store64(&report->submissions, counters.submissions);
    Store64(&report->completions, counters.completions);
    dsb(DSB_ALL);
}

}  // namespace

#if defined(PYPTO_T06_DRIVER)

T06_KERNEL_META(pypto_stage1b_t06_driver_0_mix_aiv);

namespace {

__aicore__ inline void Submit(
    __gm__ uint64_t *remote_input, __gm__ SlotControl *remote_submission, uint64_t generation,
    uint64_t sequence, uint64_t slot, uint64_t payload_words, Counters &counters
) {
    __gm__ uint64_t *payload = PayloadSlot(remote_input, slot);
    uint64_t checksum = 0;
    for (uint64_t index = 0; index < payload_words; ++index) {
        const uint64_t value = PatternWord(generation, sequence, index);
        Store64(&payload[index], value);
        checksum += value;
    }
    dsb(DSB_ALL);
    PublishMessage(
        &remote_submission[slot], generation, sequence, payload_words * sizeof(uint64_t), checksum,
        PatternWord(generation, sequence, 0), PatternWord(generation, sequence, payload_words - 1), 0, slot
    );
    ++counters.submissions;
    ++counters.credits_acquired;
}

__aicore__ inline bool ValidateCompletion(
    __gm__ uint64_t *local_output, __gm__ SlotControl *local_completion, uint64_t generation,
    uint64_t sequence, uint64_t slot, uint64_t payload_words, Counters &counters
) {
    uint64_t observed_sequence = 0;
    if (!WaitForSequence(&local_completion[slot].signal.sequence, sequence, observed_sequence)) {
        ++counters.timeouts;
        if (observed_sequence > sequence) {
            ++counters.sequence_errors;
        }
        return false;
    }
    dsb(DSB_ALL);
    __gm__ MessageLine *message = &local_completion[slot].message;
    const uint64_t completion_generation = Load64(&message->generation);
    const uint64_t completion_sequence = Load64(&message->sequence);
    const uint64_t completion_bytes = Load64(&message->payload_bytes);
    const uint64_t completion_checksum = Load64(&message->checksum);
    const uint64_t completion_head = Load64(&message->head_marker);
    const uint64_t completion_tail = Load64(&message->tail_marker);
    const uint64_t completion_status = Load64(&message->status);
    const uint64_t completion_slot = Load64(&message->slot);
    if (completion_generation != generation) {
        ++counters.generation_errors;
    }
    if (completion_sequence != sequence) {
        ++counters.sequence_errors;
    }
    if (completion_bytes != payload_words * sizeof(uint64_t) || completion_status != 0 ||
        completion_slot != slot) {
        ++counters.validation_errors;
    }

    __gm__ uint64_t *payload = PayloadSlot(local_output, slot);
    const uint64_t mask = TransformMask(sequence);
    uint64_t checksum = 0;
    uint64_t payload_mismatches = 0;
    for (uint64_t index = 0; index < payload_words; ++index) {
        const uint64_t observed = Load64(&payload[index]);
        const uint64_t expected = PatternWord(generation, sequence, index) ^ mask;
        checksum += observed;
        if (observed != expected) {
            ++payload_mismatches;
        }
    }
    if (payload_mismatches != 0 || completion_checksum != checksum) {
        ++counters.checksum_errors;
    }
    if (completion_head != (PatternWord(generation, sequence, 0) ^ mask) ||
        completion_tail != (PatternWord(generation, sequence, payload_words - 1) ^ mask)) {
        ++counters.marker_errors;
    }
    counters.validation_errors += payload_mismatches;
    ++counters.processed;
    ++counters.slot_processed[slot];
    ++counters.terminal_tasks;
    ++counters.credits_returned;
    ++counters.completions;
    ++counters.progress_after_resume;
    return true;
}

}  // namespace

extern "C" __global__ __aicore__ void pypto_stage1b_t06_driver_0_mix_aiv(
    __gm__ uint64_t *local_output, __gm__ ControlBlock *local_completion,
    __gm__ uint64_t *remote_input, __gm__ ControlBlock *remote_submission,
    uint64_t generation, uint32_t payload_words, uint32_t sequence_count
) {
    Counters counters = {};
    uint64_t available_credits = 2;
    uint64_t inflight = 0;
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());

    Submit(remote_input, remote_submission->slots, generation, 1, 0, payload_words, counters);
    --available_credits;
    ++inflight;
    Submit(remote_input, remote_submission->slots, generation, 2, 1, payload_words, counters);
    --available_credits;
    ++inflight;
    counters.max_inflight = inflight;
    counters.peak_queue_depth = inflight;

    // Sequence 3 is attempted exactly once while both credits are occupied.
    // It enters a pending state without touching either payload or descriptor.
    if (available_credits == 0) {
        ++counters.no_credit_events;
        ++counters.pending_requests;
        if (Load64(&local_completion->slots[0].signal.sequence) == 0 &&
            Load64(&local_completion->slots[1].signal.sequence) == 0) {
            counters.service_pause_observed = 1;
        } else {
            ++counters.validation_errors;
        }
        Store64(&remote_submission->backpressure.queue_depth, inflight);
        Store64(&remote_submission->backpressure.status, kNoCreditStatus);
        Store64(&remote_submission->backpressure.submission_retry_spins, counters.submission_retry_spins);
        dsb(DSB_ALL);
        Store64(&remote_submission->backpressure.pending_sequence, 3);
        dsb(DSB_ALL);
    }

    if (ValidateCompletion(local_output, local_completion->slots, generation, 1, 0, payload_words, counters)) {
        ++available_credits;
        --inflight;
        if (Load64(&local_completion->backpressure.service_resume_sequence) == 1) {
            counters.service_resume_observed = 1;
        }
        if (Load64(&local_completion->slots[0].signal.sequence) != 1) {
            ++counters.slot_overwrite_errors;
        }
        Submit(remote_input, remote_submission->slots, generation, 3, 0, payload_words, counters);
        --available_credits;
        ++inflight;
    }
    if (ValidateCompletion(local_output, local_completion->slots, generation, 2, 1, payload_words, counters)) {
        ++available_credits;
        --inflight;
        if (Load64(&local_completion->slots[1].signal.sequence) != 2) {
            ++counters.slot_overwrite_errors;
        }
        Submit(remote_input, remote_submission->slots, generation, 4, 1, payload_words, counters);
        --available_credits;
        ++inflight;
    }
    if (ValidateCompletion(local_output, local_completion->slots, generation, 3, 0, payload_words, counters)) {
        ++available_credits;
        --inflight;
    }
    if (ValidateCompletion(local_output, local_completion->slots, generation, 4, 1, payload_words, counters)) {
        ++available_credits;
        --inflight;
    }
    if (sequence_count != 4 || available_credits != 2 || inflight != 0) {
        ++counters.validation_errors;
    }

    PublishReport(&local_completion->report, counters, static_cast<uint64_t>(get_sys_cnt()) - begin);
}

#elif defined(PYPTO_T06_SERVICE)

T06_KERNEL_META(pypto_stage1b_t06_service_0_mix_aiv);

namespace {

__aicore__ inline bool Complete(
    __gm__ uint64_t *local_input, __gm__ SlotControl *local_submission,
    __gm__ uint64_t *remote_output, __gm__ SlotControl *remote_completion,
    uint64_t generation, uint64_t sequence, uint64_t slot, uint64_t payload_words, Counters &counters
) {
    uint64_t observed_sequence = 0;
    if (!WaitForSequence(&local_submission[slot].signal.sequence, sequence, observed_sequence)) {
        ++counters.timeouts;
        if (observed_sequence > sequence) {
            ++counters.sequence_errors;
        }
        return false;
    }
    dsb(DSB_ALL);
    __gm__ MessageLine *message = &local_submission[slot].message;
    const uint64_t submission_generation = Load64(&message->generation);
    const uint64_t submission_sequence = Load64(&message->sequence);
    const uint64_t submission_bytes = Load64(&message->payload_bytes);
    const uint64_t submission_checksum = Load64(&message->checksum);
    const uint64_t submission_head = Load64(&message->head_marker);
    const uint64_t submission_tail = Load64(&message->tail_marker);
    const uint64_t submission_status = Load64(&message->status);
    const uint64_t submission_slot = Load64(&message->slot);
    uint64_t task_status = 0;
    if (submission_generation != generation) {
        ++counters.generation_errors;
        ++task_status;
    }
    if (submission_sequence != sequence) {
        ++counters.sequence_errors;
        ++task_status;
    }
    if (submission_bytes != payload_words * sizeof(uint64_t) || submission_status != 0 ||
        submission_slot != slot) {
        ++counters.validation_errors;
        ++task_status;
    }

    __gm__ uint64_t *input_payload = PayloadSlot(local_input, slot);
    __gm__ uint64_t *output_payload = PayloadSlot(remote_output, slot);
    const uint64_t mask = TransformMask(sequence);
    uint64_t input_checksum = 0;
    uint64_t output_checksum = 0;
    uint64_t input_head = 0;
    uint64_t input_tail = 0;
    uint64_t output_head = 0;
    uint64_t output_tail = 0;
    uint64_t payload_mismatches = 0;
    for (uint64_t index = 0; index < payload_words; ++index) {
        const uint64_t input = Load64(&input_payload[index]);
        const uint64_t output = input ^ mask;
        if (input != PatternWord(generation, sequence, index)) {
            ++payload_mismatches;
        }
        input_checksum += input;
        output_checksum += output;
        Store64(&output_payload[index], output);
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
        ++task_status;
    }
    if (submission_head != input_head || submission_tail != input_tail) {
        ++counters.marker_errors;
        ++task_status;
    }
    counters.validation_errors += payload_mismatches;
    task_status += payload_mismatches;
    dsb(DSB_ALL);
    PublishMessage(
        &remote_completion[slot], generation, sequence, payload_words * sizeof(uint64_t), output_checksum,
        output_head, output_tail, task_status, slot
    );
    ++counters.processed;
    ++counters.slot_processed[slot];
    ++counters.terminal_tasks;
    ++counters.submissions;
    ++counters.completions;
    ++counters.progress_after_resume;
    return true;
}

}  // namespace

extern "C" __global__ __aicore__ void pypto_stage1b_t06_service_0_mix_aiv(
    __gm__ uint64_t *local_input, __gm__ ControlBlock *local_submission,
    __gm__ uint64_t *remote_output, __gm__ ControlBlock *remote_completion,
    uint64_t generation, uint32_t payload_words, uint32_t sequence_count
) {
    Counters counters = {};
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint64_t observed_pending = 0;

    // Do not consume either occupied slot until the driver records the third
    // request as pending. This makes queue saturation deterministic.
    if (!WaitForSequence(&local_submission->backpressure.pending_sequence, 3, observed_pending)) {
        ++counters.timeouts;
    } else {
        dsb(DSB_ALL);
        const uint64_t queue_depth = Load64(&local_submission->backpressure.queue_depth);
        const uint64_t status = Load64(&local_submission->backpressure.status);
        const uint64_t retry_spins = Load64(&local_submission->backpressure.submission_retry_spins);
        if (queue_depth == 2 && status == kNoCreditStatus && retry_spins == 0 &&
            Load64(&local_submission->slots[0].signal.sequence) == 1 &&
            Load64(&local_submission->slots[1].signal.sequence) == 2) {
            counters.service_pause_observed = 1;
            counters.no_credit_events = 1;
            counters.pending_requests = 1;
            counters.max_inflight = 2;
            counters.peak_queue_depth = 2;
        } else {
            ++counters.validation_errors;
        }
        Store64(&remote_completion->backpressure.service_resume_sequence, 1);
        dsb(DSB_ALL);
        counters.service_resume_observed = 1;

        if (Complete(
                local_input, local_submission->slots, remote_output, remote_completion->slots,
                generation, 1, 0, payload_words, counters)) {
            Complete(
                local_input, local_submission->slots, remote_output, remote_completion->slots,
                generation, 2, 1, payload_words, counters
            );
            Complete(
                local_input, local_submission->slots, remote_output, remote_completion->slots,
                generation, 3, 0, payload_words, counters
            );
            Complete(
                local_input, local_submission->slots, remote_output, remote_completion->slots,
                generation, 4, 1, payload_words, counters
            );
        }
    }
    if (sequence_count != 4) {
        ++counters.validation_errors;
    }

    PublishReport(&local_submission->report, counters, static_cast<uint64_t>(get_sys_cnt()) - begin);
}

#else
#error "Define PYPTO_T06_DRIVER or PYPTO_T06_SERVICE"
#endif
