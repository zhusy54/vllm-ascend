# PyPTO proxy validation

This directory contains an isolated, non-production prototype for the fixed
NPU `A -> B -> C` distributed service described in
`pypto_docs/pypto-wse-proxy-abc-validation-plan.md`.

The prototype does not import `vllm_ascend` and does not validate the PyPTO
compiler or scheduler. It reuses only the existing low-level ACL VMM and AIV
binary helpers from `tools/pypto_wse_validation`.

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

The Host only writes the request input and submission control, then polls and
reads the final result. A, remote B dispatch/completion, and C are advanced by
the two resident device kernels.

Run the complete V01-V06 matrix and create detailed gitignored artifacts:

```bash
../.venv/bin/python -m pypto_test.validation.collect_evidence \
  --attention-device 0 --wse-device 1 \
  --artifact-dir pypto_test/artifacts/full
```

The checked-in validation result is in
`pypto_docs/pypto-wse-proxy-abc-validation-record.md`.
