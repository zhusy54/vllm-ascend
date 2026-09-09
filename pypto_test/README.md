# PyPTO proxy validation

This directory contains an isolated, non-production prototype for the fixed
NPU `A -> B -> C` distributed service described in
`pypto_docs/pypto-wse-proxy-abc-validation-plan.md`.

The prototype does not import `vllm_ascend` and does not validate the PyPTO
compiler or scheduler. It reuses only the existing low-level ACL VMM, control
channel, and AIV binary helpers from `tools/pypto_wse_validation`.

Build and run instructions are added with the executable implementation.
