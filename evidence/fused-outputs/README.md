# Fused-output validation

FUSED_OUTPUT_RESULTS.md contains the measured scope, controls and limits.
The native API alone does not change kernel math or claim a serving speedup.
The vLLM prototype moves wo_a across the piecewise graph boundary; default
FULL decode already captures wo_a.

The JSON files include every output token and scheduler-step record. The driver
supports baseline/candidate and fallback-threshold selection. Adjust local paths
to the model and checkouts before use; coordinate GPU ownership first.
The comparison used TP4 GB200, fixed 64 output tokens and no prefix cache.

Source branches on foraxe/vllm:
- validation/dsv41-fused-56344: integration baseline
- research/dsv41-fused-output-prototype: measured consumer prototype
- perf/dsv41-fused-output-graph: fallback-preserving follow-up

The final vLLM follow-up depends on open PR #56344 and this native API change.
