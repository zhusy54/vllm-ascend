// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// This file is a part of the vllm-ascend project.

// Resident Attention-side driver for fixed A(NPU) -> B(remote) -> C(NPU).
//
// The Host launches this kernel once per generation.  It then remains alive
// across all synchronous requests and is the Device-side dependency driver:
//
//   Host request signal
//     -> read local_input and execute A(x) = x + 1
//     -> write A output to remote_b_input (WSE-owned VMM window)
//     -> publish remote B descriptor and submission signal
//     -> wait on local B completion written remotely by the WSE-side Device
//     -> read local_b_output and execute C(b) = b + 3
//     -> publish final output/descriptor/signal for the Host
//
// The Host never observes A output, B input/output, or the B completion.  All
// addresses are process-local VAs prepared by Bootstrap before kernel launch.
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

static const KernelMeta g_abc_driver_meta __attribute__((used, section(".ascend.meta.pypto_abc_driver_0_mix_aiv"))) = {
    {{1, sizeof(uint32_t)}, 5}, {{3, sizeof(uint32_t)}, 0, 1}};

struct alignas(64) SignalLine {
  // A signal contains only a monotonically increasing publication sequence.
  // The descriptor and payload live on separate cache lines/regions.
  uint64_t sequence;
  uint64_t reserved[7];
};

struct alignas(64) DescriptorLine {
  // The names value3..value5 keep one 64-byte wire layout usable for request,
  // task, and completion records.  Python contracts document each meaning.
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

struct alignas(64) DriverReport {
  // These counters are evidence, not inputs to scheduling.  Together they
  // prove accepted -> A -> B submission -> B completion -> C -> completed.
  uint64_t accepted;
  uint64_t completed;
  uint64_t a_runs;
  uint64_t b_submissions;
  uint64_t b_completions;
  uint64_t c_runs;
  uint64_t validation_errors;
  uint64_t generation_errors;
  uint64_t request_errors;
  uint64_t sequence_errors;
  uint64_t checksum_errors;
  uint64_t input_fences;
  uint64_t output_fences;
  uint64_t host_wait_cycles;
  uint64_t b_wait_cycles;
  uint64_t stopped;
};

static_assert(sizeof(SignalLine) == 64);
static_assert(sizeof(DescriptorLine) == 64);
static_assert(sizeof(LifecycleLine) == 64);
static_assert(sizeof(DriverReport) == 128);

__aicore__ inline uint64_t Load64(__gm__ uint64_t* address) {
  return static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
}

__aicore__ inline uint32_t Load32(__gm__ uint32_t* address) {
  return static_cast<uint32_t>(__builtin_cce_ld_dev(address, 0));
}

__aicore__ inline void Store64(__gm__ uint64_t* address, uint64_t value) { __builtin_cce_st_dev(value, address, 0); }

__aicore__ inline void Store32(__gm__ uint32_t* address, uint32_t value) { __builtin_cce_st_dev(value, address, 0); }

__aicore__ inline void PublishDescriptor(__gm__ DescriptorLine* descriptor, uint64_t generation, uint64_t request_id,
                                         uint64_t elements, uint64_t value3, uint64_t value4, uint64_t value5) {
  Store64(&descriptor->generation, generation);
  Store64(&descriptor->request_id, request_id);
  Store64(&descriptor->element_count, elements);
  Store64(&descriptor->value3, value3);
  Store64(&descriptor->value4, value4);
  Store64(&descriptor->value5, value5);
}

__aicore__ inline void PublishReport(__gm__ DriverReport* destination, const DriverReport& report) {
  // Reports are updated after each request and once more at STOP.  Host reads
  // the final snapshot only after the resident stream has synchronized.
  __gm__ uint64_t* output = reinterpret_cast<__gm__ uint64_t*>(destination);
  const uint64_t* input = reinterpret_cast<const uint64_t*>(&report);
  for (uint64_t index = 0; index < 16; ++index) {
    Store64(&output[index], input[index]);
  }
  dsb(DSB_ALL);
}

}  // namespace

extern "C" __global__ __aicore__ void pypto_abc_driver_0_mix_aiv(
    __gm__ uint32_t* local_input, __gm__ uint32_t* local_b_output, __gm__ uint32_t* local_final_output,
    __gm__ SignalLine* host_request_signal, __gm__ DescriptorLine* host_request_descriptor,
    __gm__ SignalLine* host_result_signal, __gm__ DescriptorLine* host_result_descriptor,
    __gm__ SignalLine* b_completion_signal, __gm__ DescriptorLine* b_completion_descriptor,
    __gm__ LifecycleLine* lifecycle, __gm__ DriverReport* report_address, __gm__ uint32_t* remote_b_input,
    __gm__ SignalLine* remote_submission_signal, __gm__ DescriptorLine* remote_submission_descriptor,
    uint64_t generation, uint64_t max_elements) {
  DriverReport report = {};
  uint64_t last_sequence = 0;
  // READY is published by the Device, so service admission begins only after
  // the resident loop and all bound addresses are actually usable.
  Store64(&lifecycle->ready, 1);
  dsb(DSB_ALL);

  // STOP is checked both while idle and while waiting for B.  The first-version
  // normal drain calls STOP only when no request is in flight.
  while (Load64(&lifecycle->stop_requested) == 0) {
    ++report.host_wait_cycles;
    const uint64_t sequence = Load64(&host_request_signal->sequence);
    if (sequence <= last_sequence) {
      continue;
    }
    // Host writes input and descriptor before request signal.  The fence after
    // observing a new signal orders subsequent descriptor/payload reads.
    dsb(DSB_ALL);
    const uint64_t request_generation = Load64(&host_request_descriptor->generation);
    const uint64_t request_id = Load64(&host_request_descriptor->request_id);
    const uint64_t elements = Load64(&host_request_descriptor->element_count);
    const uint64_t input_checksum = Load64(&host_request_descriptor->value3);
    const uint64_t descriptor_sequence = Load64(&host_request_descriptor->value4);
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

    // Stage A executes locally, but its output destination is an imported
    // mapping of WSE-side HBM.  Store32 therefore performs the NPU->WSE
    // Device data transfer without a Host copy.
    uint64_t observed_input_checksum = 0;
    uint64_t a_checksum = 0;
    for (uint64_t index = 0; index < elements; ++index) {
      const uint32_t input = Load32(&local_input[index]);
      const uint32_t a_output = input + 1U;
      observed_input_checksum += input;
      a_checksum += a_output;
      Store32(&remote_b_input[index], a_output);
    }
    ++report.a_runs;
    if (observed_input_checksum != input_checksum) {
      ++report.checksum_errors;
    }
    // Publication rule for A->B:
    //   remote payload -> fence -> remote descriptor -> fence -> remote signal.
    // B treats the signal as ownership of a complete, visible submission.
    dsb(DSB_ALL);
    ++report.output_fences;
    PublishDescriptor(remote_submission_descriptor, generation, request_id, elements, a_checksum, sequence, 0);
    dsb(DSB_ALL);
    Store64(&remote_submission_signal->sequence, sequence);
    dsb(DSB_ALL);
    ++report.b_submissions;

    // This is the A->B->C dependency edge.  The Attention Device waits on its
    // local completion cache line, but that line is written remotely by the
    // WSE-side Device.  No Host RPC or polling participates here.
    while (Load64(&b_completion_signal->sequence) < sequence && Load64(&lifecycle->stop_requested) == 0) {
      ++report.b_wait_cycles;
    }
    if (Load64(&lifecycle->stop_requested) != 0) {
      break;
    }
    // B publishes output and descriptor before its completion signal.  Fence
    // before consuming them to preserve the matching visibility order.
    dsb(DSB_ALL);
    ++report.input_fences;
    ++report.b_completions;
    const uint64_t completion_generation = Load64(&b_completion_descriptor->generation);
    const uint64_t completion_request = Load64(&b_completion_descriptor->request_id);
    const uint64_t completion_elements = Load64(&b_completion_descriptor->element_count);
    const uint64_t completion_status = Load64(&b_completion_descriptor->value3);
    const uint64_t completion_checksum = Load64(&b_completion_descriptor->value4);
    const uint64_t completion_sequence = Load64(&b_completion_descriptor->value5);
    if (completion_generation != generation) {
      ++report.generation_errors;
    }
    if (completion_request != request_id) {
      ++report.request_errors;
    }
    if (completion_elements != elements || completion_status != 0) {
      ++report.validation_errors;
    }
    if (completion_sequence != sequence) {
      ++report.sequence_errors;
    }

    // Stage C consumes the B output already present in Attention-owned HBM.
    // uint32 arithmetic deliberately wraps, matching the Python CPU oracle.
    uint64_t observed_b_checksum = 0;
    uint64_t final_checksum = 0;
    for (uint64_t index = 0; index < elements; ++index) {
      const uint32_t b_output = Load32(&local_b_output[index]);
      const uint32_t final_output = b_output + 3U;
      observed_b_checksum += b_output;
      final_checksum += final_output;
      Store32(&local_final_output[index], final_output);
    }
    ++report.c_runs;
    if (observed_b_checksum != completion_checksum) {
      ++report.checksum_errors;
    }
    const uint64_t status = report.validation_errors + report.generation_errors + report.request_errors +
                            report.sequence_errors + report.checksum_errors;
    // Final publication rule mirrors A->B.  Only this final signal is polled by
    // Host: C output -> fence -> completion descriptor -> fence -> signal.
    dsb(DSB_ALL);
    ++report.output_fences;
    PublishDescriptor(host_result_descriptor, generation, request_id, elements, status, final_checksum, sequence);
    dsb(DSB_ALL);
    Store64(&host_result_signal->sequence, sequence);
    dsb(DSB_ALL);
    ++report.completed;
    last_sequence = sequence;
    PublishReport(report_address, report);
  }
  // STOP acknowledgement is written while mappings are still valid.  The Host
  // synchronizes this stream before Bootstrap is allowed to unmap memory.
  report.stopped = 1;
  Store64(&lifecycle->stopped, 1);
  PublishReport(report_address, report);
}
