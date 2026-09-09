# PyPTO proxy validation

This directory contains an isolated, non-production prototype for the fixed
NPU `A -> B -> C` distributed service described in
`pypto_docs/pypto-wse-proxy-abc-validation-plan.md`.

The prototype does not import `vllm_ascend` and does not validate the PyPTO
compiler or scheduler. Code is separated by ownership:

- `pseudo_pypto/` owns the service API, local Device memory, H2D/D2H, resident
  kernels, and the fixed Device communication ABI.
- `infrastructure/` owns Host RPC plus ACL/VMM allocation, handle exchange,
  peer mapping, and release. It injects only process-local shared addresses.
- `validation/` owns deterministic inputs, the CPU oracle, and evidence checks.

The current infrastructure provider reuses only the low-level
`AclVmmRuntime` from `tools/pypto_wse_validation`; pseudo-PyPTO does not import
that helper.

Build the two resident kernels:

```bash
bash pypto_test/build_kernels.sh
```

Run one proxy-service instance directly with two same-host NPU devices:

```bash
../.venv/bin/python -m pypto_test.run_proxy_service \
  --attention-device 0 --wse-device 1 \
  --elements 1024 --start-order attention-first
```

PyPTO writes the request input to local NPU memory, then polls and reads only
the final C result. A, remote B dispatch/completion, and C are advanced by the
two resident Device kernels; Host RPC carries no per-request message.

Run the complete V01-V06 matrix and create detailed gitignored artifacts:

```bash
../.venv/bin/python -m pypto_test.validation.collect_evidence \
  --attention-device 0 --wse-device 1 \
  --artifact-dir pypto_test/artifacts/full
```

The checked-in validation result is in
`pypto_docs/pypto-wse-proxy-abc-validation-record.md`.
