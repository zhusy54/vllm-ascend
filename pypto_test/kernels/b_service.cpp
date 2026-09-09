// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// This file is a part of the vllm-ascend project.

// Resident second-NPU implementation of the WSE-side fixed B task.
//
// This kernel validates the interface expected from a future WSE backend while
// still running on a second Ascend NPU.  It is launched once per generation:
//
//   wait on local submission written by the Attention Device
//     -> consume A output from local_b_input
//     -> execute B(a) = 2 * a
//     -> write B output to remote_b_output (Attention-owned VMM window)
//     -> publish B completion directly to the Attention Device
//
// The WSE Host handles only lifecycle RPC.  It does not receive an
// execute call, copy A/B payloads, or forward the completion.
#include "cce_aicore_intrinsics.h"
#include <stdint.h>

namespace {

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

static const KernelMeta g_b_service_meta __attribute__((used, section(".ascend.meta.pypto_b_service_0_mix_aiv"))) = {
    {{1, sizeof(uint32_t)}, 5}, {{3, sizeof(uint32_t)}, 0, 1}};

struct alignas(64) SignalLine {
  // Descriptor/payload publication completes before this sequence changes.
  uint64_t sequence;
  uint64_t reserved[7];
};
struct alignas(64) DescriptorLine {
  uint64_t generation;
  uint64_t request_id;
  uint64_t element_count;
  uint64_t value3;
  uint64_t value4;
  uint64_t value5;
  uint64_t reserved[2];
};
struct alignas(64) LifecycleLine {
  uint64_t stop_requested;
  uint64_t stopped;
  uint64_t ready;
  uint64_t reserved[5];
};
struct alignas(64) ServiceReport {
  // Final evidence that the resident service accepted and ran B for every
  // sequence, with no Host-side task invocation.
  uint64_t accepted;
  uint64_t completed;
  uint64_t b_runs;
  uint64_t validation_errors;
  uint64_t generation_errors;
  uint64_t request_errors;
  uint64_t sequence_errors;
  uint64_t checksum_errors;
  uint64_t input_fences;
  uint64_t output_fences;
  uint64_t submission_wait_cycles;
  uint64_t stopped;
  uint64_t reserved[4];
};

static_assert(sizeof(SignalLine) == 64);
static_assert(sizeof(DescriptorLine) == 64);
static_assert(sizeof(LifecycleLine) == 64);
static_assert(sizeof(ServiceReport) == 128);

__aicore__ inline uint64_t Load64(__gm__ uint64_t* address) {
  return static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
}
__aicore__ inline uint32_t Load32(__gm__ uint32_t* address) {
  return static_cast<uint32_t>(__builtin_cce_ld_dev(address, 0));
}
__aicore__ inline void Store64(__gm__ uint64_t* address, uint64_t value) { __builtin_cce_st_dev(value, address, 0); }
__aicore__ inline void Store32(__gm__ uint32_t* address, uint32_t value) { __builtin_cce_st_dev(value, address, 0); }
__aicore__ inline void PublishReport(__gm__ ServiceReport* destination, const ServiceReport& report) {
  // Host reads this report only after STOP and stream synchronization.
  __gm__ uint64_t* output = reinterpret_cast<__gm__ uint64_t*>(destination);
  const uint64_t* input = reinterpret_cast<const uint64_t*>(&report);
  for (uint64_t index = 0; index < 16; ++index) {
    Store64(&output[index], input[index]);
  }
  dsb(DSB_ALL);
}

}  // namespace

extern "C" __global__ __aicore__ void pypto_b_service_0_mix_aiv(
    __gm__ uint32_t* local_b_input, __gm__ SignalLine* submission_signal, __gm__ DescriptorLine* submission_descriptor,
    __gm__ LifecycleLine* lifecycle, __gm__ ServiceReport* report_address, __gm__ uint32_t* remote_b_output,
    __gm__ SignalLine* remote_completion_signal, __gm__ DescriptorLine* remote_completion_descriptor,
    uint64_t generation, uint64_t max_elements) {
  ServiceReport report = {};
  uint64_t last_sequence = 0;
  // Device-published READY confirms that the persistent submission loop is live.
  Store64(&lifecycle->ready, 1);
  dsb(DSB_ALL);

  // Submission signals arrive through remote stores from the Attention NPU,
  // not through this process's TCP control loop.
  while (Load64(&lifecycle->stop_requested) == 0) {
    ++report.submission_wait_cycles;
    const uint64_t sequence = Load64(&submission_signal->sequence);
    if (sequence <= last_sequence) {
      continue;
    }
    // Attention published A payload and descriptor before the signal.  Fence
    // after observing it before consuming either region.
    dsb(DSB_ALL);
    ++report.input_fences;
    const uint64_t request_generation = Load64(&submission_descriptor->generation);
    const uint64_t request_id = Load64(&submission_descriptor->request_id);
    const uint64_t elements = Load64(&submission_descriptor->element_count);
    const uint64_t expected_checksum = Load64(&submission_descriptor->value3);
    const uint64_t descriptor_sequence = Load64(&submission_descriptor->value4);
    if (request_generation != generation) {
      ++report.generation_errors;
    }
    if (request_id == 0) {
      ++report.request_errors;
    }
    if (descriptor_sequence != sequence) {
      ++report.sequence_errors;
    }
    if (elements == 0 || elements > max_elements) {
      ++report.validation_errors;
      continue;
    }
    ++report.accepted;

    // Stage B reads WSE-owned local HBM and writes its output through an
    // imported mapping of Attention HBM.  This Store32 loop is the WSE-side
    // ->NPU Device data path being validated.
    uint64_t input_checksum = 0;
    uint64_t output_checksum = 0;
    for (uint64_t index = 0; index < elements; ++index) {
      const uint32_t input = Load32(&local_b_input[index]);
      const uint32_t output = input * 2U;
      input_checksum += input;
      output_checksum += output;
      Store32(&remote_b_output[index], output);
    }
    ++report.b_runs;
    if (input_checksum != expected_checksum) {
      ++report.checksum_errors;
    }
    const uint64_t status = report.validation_errors + report.generation_errors + report.request_errors +
                            report.sequence_errors + report.checksum_errors;
    // Publication rule for B->C:
    //   remote B payload -> fence -> remote completion descriptor
    //   -> fence -> remote completion signal.
    // The Attention driver, rather than Host, consumes this notification.
    dsb(DSB_ALL);
    ++report.output_fences;
    Store64(&remote_completion_descriptor->generation, generation);
    Store64(&remote_completion_descriptor->request_id, request_id);
    Store64(&remote_completion_descriptor->element_count, elements);
    Store64(&remote_completion_descriptor->value3, status);
    Store64(&remote_completion_descriptor->value4, output_checksum);
    Store64(&remote_completion_descriptor->value5, sequence);
    dsb(DSB_ALL);
    Store64(&remote_completion_signal->sequence, sequence);
    dsb(DSB_ALL);
    ++report.completed;
    last_sequence = sequence;
    PublishReport(report_address, report);
  }
  // Acknowledge STOP before the backend synchronizes/unloads this kernel and
  // before Bootstrap tears down either VMM mapping.
  report.stopped = 1;
  Store64(&lifecycle->stopped, 1);
  PublishReport(report_address, report);
}
