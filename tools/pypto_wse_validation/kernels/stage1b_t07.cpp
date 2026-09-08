// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// This file is a part of the vllm-ascend project.

// Pure CCEC AIV kernels for Stage 1B T07. Each sequence transfers a full 1 MiB
// payload with sequence-unique head/tail markers. The driver validates output
// immediately after observing completion and never inserts a compensating wait.
#include "cce_aicore_intrinsics.h"
#include <stdint.h>

namespace {

constexpr uint64_t kWaitTimeoutCycles = 40000000000ULL;
constexpr uint64_t kPatternMagic = 0x505950544f543037ULL;
constexpr uint64_t kHeadMarkerMagic = 0x4845414454303700ULL;
constexpr uint64_t kTailMarkerMagic = 0x5441494c54303700ULL;
constexpr uint64_t kGenerationMix = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t kSequenceMix = 0xd1b54a32d192ed03ULL;
constexpr uint64_t kIndexMix = 0x94d049bb133111ebULL;
constexpr uint64_t kByteRepeat = 0x0101010101010101ULL;
constexpr uint64_t kSlotStrideWords = (1024ULL * 1024ULL) / sizeof(uint64_t);

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

#define T07_KERNEL_META(kernel_name)                                                                        \
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

struct alignas(64) ReportBlock {
    uint64_t processed;
    uint64_t validation_errors;
    uint64_t sequence_errors;
    uint64_t generation_errors;
    uint64_t checksum_errors;
    uint64_t head_marker_errors;
    uint64_t tail_marker_errors;
    uint64_t stale_payload_errors;
    uint64_t incomplete_payload_errors;
    uint64_t premature_completion_errors;
    uint64_t timeouts;
    uint64_t elapsed_cycles;
    uint64_t slot0_processed;
    uint64_t slot1_processed;
    uint64_t input_publish_fences;
    uint64_t input_visibility_checks;
    uint64_t output_publish_fences;
    uint64_t immediate_completion_checks;
    uint64_t post_completion_delay_cycles;
    uint64_t submissions;
    uint64_t completions;
    uint64_t slot_overwrite_errors;
    uint64_t payload_words_validated;
    uint64_t unique_tail_markers_validated;
    uint64_t max_inflight;
    uint64_t reserved;
};

struct alignas(64) ControlBlock {
    SlotControl slots[2];
    ReportBlock report;
};

struct Counters {
    uint64_t processed;
    uint64_t validation_errors;
    uint64_t sequence_errors;
    uint64_t generation_errors;
    uint64_t checksum_errors;
    uint64_t head_marker_errors;
    uint64_t tail_marker_errors;
    uint64_t stale_payload_errors;
    uint64_t incomplete_payload_errors;
    uint64_t premature_completion_errors;
    uint64_t timeouts;
    uint64_t slot_processed[2];
    uint64_t input_publish_fences;
    uint64_t input_visibility_checks;
    uint64_t output_publish_fences;
    uint64_t immediate_completion_checks;
    uint64_t post_completion_delay_cycles;
    uint64_t submissions;
    uint64_t completions;
    uint64_t slot_overwrite_errors;
    uint64_t payload_words_validated;
    uint64_t unique_tail_markers_validated;
    uint64_t max_inflight;
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

__aicore__ inline uint64_t HeadMarker(uint64_t generation, uint64_t sequence) {
    return kHeadMarkerMagic ^ (generation * kGenerationMix) ^ (sequence * kSequenceMix);
}

__aicore__ inline uint64_t TailMarker(uint64_t generation, uint64_t sequence, uint64_t payload_words) {
    return kTailMarkerMagic ^ (generation * kGenerationMix) ^ (sequence * kSequenceMix) ^
        (payload_words * kIndexMix);
}

__aicore__ inline uint64_t PayloadWord(
    uint64_t generation, uint64_t sequence, uint64_t index, uint64_t payload_words
) {
    if (index == 0) {
        return HeadMarker(generation, sequence);
    }
    if (index + 1 == payload_words) {
        return TailMarker(generation, sequence, payload_words);
    }
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
        counters.generation_errors + counters.checksum_errors + counters.head_marker_errors +
        counters.tail_marker_errors + counters.stale_payload_errors + counters.incomplete_payload_errors +
        counters.premature_completion_errors + counters.timeouts + counters.slot_overwrite_errors);
    Store64(&report->sequence_errors, counters.sequence_errors);
    Store64(&report->generation_errors, counters.generation_errors);
    Store64(&report->checksum_errors, counters.checksum_errors);
    Store64(&report->head_marker_errors, counters.head_marker_errors);
    Store64(&report->tail_marker_errors, counters.tail_marker_errors);
    Store64(&report->stale_payload_errors, counters.stale_payload_errors);
    Store64(&report->incomplete_payload_errors, counters.incomplete_payload_errors);
    Store64(&report->premature_completion_errors, counters.premature_completion_errors);
    Store64(&report->timeouts, counters.timeouts);
    Store64(&report->elapsed_cycles, elapsed_cycles);
    Store64(&report->slot0_processed, counters.slot_processed[0]);
    Store64(&report->slot1_processed, counters.slot_processed[1]);
    Store64(&report->input_publish_fences, counters.input_publish_fences);
    Store64(&report->input_visibility_checks, counters.input_visibility_checks);
    Store64(&report->output_publish_fences, counters.output_publish_fences);
    Store64(&report->immediate_completion_checks, counters.immediate_completion_checks);
    Store64(&report->post_completion_delay_cycles, counters.post_completion_delay_cycles);
    Store64(&report->submissions, counters.submissions);
    Store64(&report->completions, counters.completions);
    Store64(&report->slot_overwrite_errors, counters.slot_overwrite_errors);
    Store64(&report->payload_words_validated, counters.payload_words_validated);
    Store64(&report->unique_tail_markers_validated, counters.unique_tail_markers_validated);
    Store64(&report->max_inflight, counters.max_inflight);
    Store64(&report->reserved, 0);
    dsb(DSB_ALL);
}

}  // namespace

#if defined(PYPTO_T07_DRIVER)

T07_KERNEL_META(pypto_stage1b_t07_driver_0_mix_aiv);

extern "C" __global__ __aicore__ void pypto_stage1b_t07_driver_0_mix_aiv(
    __gm__ uint64_t *local_output, __gm__ ControlBlock *local_completion,
    __gm__ uint64_t *remote_input, __gm__ ControlBlock *remote_submission,
    uint64_t generation, uint32_t payload_words, uint32_t sequence_count
) {
    Counters counters = {};
    counters.max_inflight = 1;
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());

    for (uint64_t sequence = 1; sequence <= sequence_count; ++sequence) {
        const uint64_t slot = (sequence - 1) & 1ULL;
        if (sequence > 2 && Load64(&local_completion->slots[slot].signal.sequence) != sequence - 2) {
            ++counters.slot_overwrite_errors;
        }
        __gm__ uint64_t *input = PayloadSlot(remote_input, slot);
        uint64_t input_checksum = 0;
        for (uint64_t index = 0; index < payload_words; ++index) {
            const uint64_t value = PayloadWord(generation, sequence, index, payload_words);
            Store64(&input[index], value);
            input_checksum += value;
        }
        dsb(DSB_ALL);
        ++counters.input_publish_fences;
        PublishMessage(
            &remote_submission->slots[slot], generation, sequence, payload_words * sizeof(uint64_t),
            input_checksum, HeadMarker(generation, sequence), TailMarker(generation, sequence, payload_words), 0,
            slot
        );
        ++counters.submissions;

        uint64_t observed_sequence = 0;
        if (!WaitForSequence(
                &local_completion->slots[slot].signal.sequence, sequence, observed_sequence)) {
            ++counters.timeouts;
            if (observed_sequence > sequence) {
                ++counters.sequence_errors;
            }
            break;
        }
        // There is deliberately no wait or delay between observing the signal
        // and reading completion metadata plus the entire output payload.
        dsb(DSB_ALL);
        ++counters.immediate_completion_checks;

        __gm__ MessageLine *message = &local_completion->slots[slot].message;
        const uint64_t completion_generation = Load64(&message->generation);
        const uint64_t completion_sequence = Load64(&message->sequence);
        const uint64_t completion_bytes = Load64(&message->payload_bytes);
        const uint64_t completion_checksum = Load64(&message->checksum);
        const uint64_t completion_head = Load64(&message->head_marker);
        const uint64_t completion_tail = Load64(&message->tail_marker);
        const uint64_t completion_status = Load64(&message->status);
        const uint64_t completion_slot = Load64(&message->slot);
        uint64_t task_visibility_errors = 0;
        if (completion_generation != generation) {
            ++counters.generation_errors;
            ++task_visibility_errors;
        }
        if (completion_sequence != sequence) {
            ++counters.sequence_errors;
            ++task_visibility_errors;
        }
        if (completion_bytes != payload_words * sizeof(uint64_t) || completion_status != 0 ||
            completion_slot != slot) {
            ++counters.validation_errors;
            ++task_visibility_errors;
        }

        __gm__ uint64_t *output = PayloadSlot(local_output, slot);
        const uint64_t mask = TransformMask(sequence);
        uint64_t output_checksum = 0;
        uint64_t payload_mismatches = 0;
        for (uint64_t index = 0; index < payload_words; ++index) {
            const uint64_t observed = Load64(&output[index]);
            const uint64_t expected = PayloadWord(generation, sequence, index, payload_words) ^ mask;
            output_checksum += observed;
            if (observed != expected) {
                ++payload_mismatches;
            }
        }
        if (payload_mismatches != 0 || completion_checksum != output_checksum) {
            ++counters.checksum_errors;
            ++task_visibility_errors;
        }
        if (completion_head != (HeadMarker(generation, sequence) ^ mask)) {
            ++counters.head_marker_errors;
            ++task_visibility_errors;
        }
        if (completion_tail != (TailMarker(generation, sequence, payload_words) ^ mask)) {
            ++counters.tail_marker_errors;
            ++task_visibility_errors;
        } else {
            ++counters.unique_tail_markers_validated;
        }
        if (task_visibility_errors != 0 || payload_mismatches != 0) {
            ++counters.premature_completion_errors;
        }
        counters.validation_errors += payload_mismatches;
        counters.payload_words_validated += payload_words;
        ++counters.processed;
        ++counters.slot_processed[slot];
        ++counters.completions;
    }

    PublishReport(&local_completion->report, counters, static_cast<uint64_t>(get_sys_cnt()) - begin);
}

#elif defined(PYPTO_T07_SERVICE)

T07_KERNEL_META(pypto_stage1b_t07_service_0_mix_aiv);

extern "C" __global__ __aicore__ void pypto_stage1b_t07_service_0_mix_aiv(
    __gm__ uint64_t *local_input, __gm__ ControlBlock *local_submission,
    __gm__ uint64_t *remote_output, __gm__ ControlBlock *remote_completion,
    uint64_t generation, uint32_t payload_words, uint32_t sequence_count
) {
    Counters counters = {};
    counters.max_inflight = 1;
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());

    for (uint64_t sequence = 1; sequence <= sequence_count; ++sequence) {
        const uint64_t slot = (sequence - 1) & 1ULL;
        uint64_t observed_sequence = 0;
        if (!WaitForSequence(
                &local_submission->slots[slot].signal.sequence, sequence, observed_sequence)) {
            ++counters.timeouts;
            if (observed_sequence > sequence) {
                ++counters.sequence_errors;
            }
            break;
        }
        dsb(DSB_ALL);
        ++counters.input_visibility_checks;

        __gm__ MessageLine *message = &local_submission->slots[slot].message;
        const uint64_t submission_generation = Load64(&message->generation);
        const uint64_t submission_sequence = Load64(&message->sequence);
        const uint64_t submission_bytes = Load64(&message->payload_bytes);
        const uint64_t submission_checksum = Load64(&message->checksum);
        const uint64_t submission_head = Load64(&message->head_marker);
        const uint64_t submission_tail = Load64(&message->tail_marker);
        const uint64_t submission_status = Load64(&message->status);
        const uint64_t submission_slot = Load64(&message->slot);
        uint64_t task_status = 0;
        uint64_t task_incomplete = 0;
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

        __gm__ uint64_t *input = PayloadSlot(local_input, slot);
        __gm__ uint64_t *output = PayloadSlot(remote_output, slot);
        const uint64_t mask = TransformMask(sequence);
        uint64_t input_checksum = 0;
        uint64_t output_checksum = 0;
        uint64_t input_head = 0;
        uint64_t input_tail = 0;
        uint64_t output_head = 0;
        uint64_t output_tail = 0;
        uint64_t payload_mismatches = 0;
        uint64_t stale_words = 0;
        for (uint64_t index = 0; index < payload_words; ++index) {
            const uint64_t observed = Load64(&input[index]);
            const uint64_t expected = PayloadWord(generation, sequence, index, payload_words);
            const uint64_t transformed = observed ^ mask;
            if (observed != expected) {
                ++payload_mismatches;
                if (sequence > 2 &&
                    observed == PayloadWord(generation, sequence - 2, index, payload_words)) {
                    ++stale_words;
                }
            }
            input_checksum += observed;
            output_checksum += transformed;
            Store64(&output[index], transformed);
            if (index == 0) {
                input_head = observed;
                output_head = transformed;
            }
            if (index + 1 == payload_words) {
                input_tail = observed;
                output_tail = transformed;
            }
        }
        if (submission_checksum != input_checksum) {
            ++counters.checksum_errors;
            ++task_status;
            ++task_incomplete;
        }
        if (submission_head != input_head || input_head != HeadMarker(generation, sequence)) {
            ++counters.head_marker_errors;
            ++task_status;
            ++task_incomplete;
        }
        if (submission_tail != input_tail || input_tail != TailMarker(generation, sequence, payload_words)) {
            ++counters.tail_marker_errors;
            ++task_status;
            ++task_incomplete;
        } else {
            ++counters.unique_tail_markers_validated;
        }
        if (stale_words != 0) {
            ++counters.stale_payload_errors;
            ++task_status;
        }
        if (payload_mismatches != 0) {
            ++task_incomplete;
            task_status += payload_mismatches;
        }
        if (task_incomplete != 0) {
            ++counters.incomplete_payload_errors;
        }
        counters.validation_errors += payload_mismatches;
        counters.payload_words_validated += payload_words;
        ++counters.submissions;
        dsb(DSB_ALL);
        ++counters.output_publish_fences;
        PublishMessage(
            &remote_completion->slots[slot], generation, sequence, payload_words * sizeof(uint64_t),
            output_checksum, output_head, output_tail, task_status, slot
        );
        ++counters.processed;
        ++counters.slot_processed[slot];
        ++counters.completions;
    }

    PublishReport(&local_submission->report, counters, static_cast<uint64_t>(get_sys_cnt()) - begin);
}

#else
#error "Define PYPTO_T07_DRIVER or PYPTO_T07_SERVICE"
#endif
