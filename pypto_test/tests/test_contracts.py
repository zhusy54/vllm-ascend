# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Public API, Device ABI, generation, and lease unit tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pypto_test.pseudo_pypto.api import ExecuteRequest, PyptoDistributedService
from pypto_test.pseudo_pypto.communication import (
    ABC_TASKS,
    CACHE_LINE_BYTES,
    DEFAULT_LAYOUT,
    EXPECTED_LAYOUT_HASH,
    MAX_ELEMENTS,
    NPU_LOCAL_WINDOW_BYTES,
    NPU_SHARED_WINDOW_BYTES,
    WSE_LOCAL_WINDOW_BYTES,
    WSE_SHARED_WINDOW_BYTES,
    BootstrapLease,
    CompletionDescriptor,
    ContractError,
    DriverReport,
    EndpointBundle,
    HostRequestDescriptor,
    LeaseState,
    LifecycleLine,
    NpuCommunicationBinding,
    PseudoProgramSpec,
    RemoteTaskDescriptor,
    ServiceReport,
    SignalLine,
    checksum_u32,
)
from pypto_test.validation.validation_utils import expected_abc, get_input_payload


def test_fixed_program_and_split_memory_layout_are_stable():
    spec = PseudoProgramSpec()
    spec.validate()
    assert spec.tasks == ABC_TASKS
    assert DEFAULT_LAYOUT.layout_hash == EXPECTED_LAYOUT_HASH
    assert NPU_LOCAL_WINDOW_BYTES > 2 * 1024 * 1024
    assert NPU_SHARED_WINDOW_BYTES > 1024 * 1024
    assert WSE_SHARED_WINDOW_BYTES > 1024 * 1024
    assert WSE_LOCAL_WINDOW_BYTES < 1024


@pytest.mark.parametrize(
    "spec",
    (
        PseudoProgramSpec(program_id="other"),
        PseudoProgramSpec(dtype="float32"),
        PseudoProgramSpec(max_elements=MAX_ELEMENTS - 1),
        PseudoProgramSpec(tasks=ABC_TASKS[:2]),
    ),
)
def test_program_rejects_non_abc_contracts(spec):
    with pytest.raises(ContractError):
        spec.validate()


def test_control_descriptors_round_trip_as_cache_lines():
    values = (
        (HostRequestDescriptor(1, 2, 3, 4, 5), HostRequestDescriptor.from_bytes),
        (RemoteTaskDescriptor(1, 2, 3, 4, 5), RemoteTaskDescriptor.from_bytes),
        (CompletionDescriptor(1, 2, 3, 0, 4, 5), CompletionDescriptor.from_bytes),
        (SignalLine(5), SignalLine.from_bytes),
        (LifecycleLine(stop_requested=1, ready=1), LifecycleLine.from_bytes),
    )
    for value, decoder in values:
        payload = value.to_bytes()
        assert len(payload) == CACHE_LINE_BYTES
        assert decoder(payload) == value


def test_descriptor_rejects_invalid_element_count():
    with pytest.raises(ContractError, match="element_count"):
        replace(HostRequestDescriptor(1, 1, 1, 0, 1), element_count=MAX_ELEMENTS + 1).to_bytes()


def test_uint32_reference_is_deterministic_and_wraps():
    payload = get_input_payload(generation=3, request_id=7, element_count=32)
    assert payload == get_input_payload(generation=3, request_id=7, element_count=32)
    assert payload != get_input_payload(generation=3, request_id=8, element_count=32)
    assert len(expected_abc(payload)) == len(payload)
    wrap = bytes.fromhex("ffffffff")
    assert expected_abc(wrap) == bytes.fromhex("03000000")
    assert checksum_u32(wrap) == 0xFFFFFFFF


def test_device_report_wire_sizes_are_stable():
    payload = bytes(16 * 8)
    assert DriverReport.from_bytes(payload).accepted == 0
    assert ServiceReport.from_bytes(payload).completed == 0


def test_endpoint_bundle_accepts_addresses_but_no_allocator_or_copy_capability():
    lease = BootstrapLease("lease", 1)
    lease.borrow()
    binding = NpuCommunicationBinding(1, 1, NPU_SHARED_WINDOW_BYTES, 2, WSE_SHARED_WINDOW_BYTES)
    bundle = EndpointBundle(1, "attention", "WSE", "FAKE", "HOST_LOCAL", DEFAULT_LAYOUT, binding, object(), lease)
    bundle.validate()
    assert not hasattr(bundle, "execution_port")
    assert not hasattr(bundle, "memory_provider")
    lease.quiesce()
    lease.release()
    assert lease.state is LeaseState.RELEASED
    with pytest.raises(ContractError, match="lease"):
        bundle.validate()


def test_service_api_is_typed_and_runtime_checkable():
    assert ExecuteRequest(bytes(4)).payload == bytes(4)
    assert PyptoDistributedService is not None
