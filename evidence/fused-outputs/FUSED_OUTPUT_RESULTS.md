# DeepSeek V4.1 fused-output ownership

## Result and scope

Caller-owned FP8 values and packed UE8M0 scales allow the fused attention result
to cross an eager break, with wo_a and wo_b captured afterwards. Current vLLM
already captures wo_a in FULL decode graphs, so this is a piecewise-prefill
optimization, not a general decode graph-boundary fix.

The FlashMLA API extension passes 27 output-buffer tests on GB200: allocating
versus provided outputs, independent optional destinations, sliced/padded scale
buffers, neighboring-row guards, invalid arguments, CUDA graph replay, and an
FP4 extra cache. No kernel math changes.

## Matched model comparison

TP4 GB200, official DeepSeek-V4.1-Flash at dba1be0a40aa45a94ad051997016db3960a90277.
V1 runner, fused attention, FULL_AND_PIECEWISE, graph sizes [1,2,4,32,64], no
prefix cache, 1 GiB KV per worker, batch/concurrency 1. 64 generated tokens with
ignore_eos=True. Context lengths 17, 8192 and 32768; 1, 1 and 4 prefill steps,
then 63 nonempty decode steps, verified on all workers. Zero-token cleanup calls
are tracked separately. Two warmups plus six measured requests per context/run.
Four separately loaded engines in A/B/B/A order; 12 measured requests per arm
and context. Diagnostic projection/replay hooks are removed before timing.

| Input tokens | Baseline TTFT ms | Candidate TTFT ms | Change | Baseline TPOT ms | Candidate TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 41.847 | 36.924 | -11.76% | 6.4289 | 6.4284 |
| 8192 | 165.853 | 163.468 | -1.44% | 6.9675 | 6.9820 |
| 32768 | 665.456 | 657.151 | -1.25% | 7.0284 | 7.0348 |

These are pooled medians of measured requests. Short-input TTFT improved in both
order comparisons: A1/B1 40.892/36.635 ms and A2/B2 41.984/37.087 ms. The long-input
results were order-sensitive and are not promoted as speedups. TPOT stayed within
0.21%; no decode speedup is claimed. This is bounded local evidence, not a broad
workload or statistical significance claim.

All 96 request outputs match token-for-token across arms. Mixed prefill/decode
outputs also match. The model diagnostic answered 323 correctly; for its actual
piecewise replay, baseline made 40 eager wo_a calls and candidate made zero.
Decode replay used the full graph in both cases.

## Revisions and implementation

- Main base: 22258a26bc090bccf5473cf681bbe9bac41bd035.
- Integration baseline: 6ff1e20b4f51729d84419f6f32f40c09b9e1ca74, incorporating open
  vLLM PR #56344 (head 8c2d92c) and preserving current JIT-warmup ownership.
- Measured consumer prototype: 74d2d52 (six-file incremental change).
- FlashMLA base: c112cc1ed1c61bfdb7dbf25e53753bd272edb767.
- Native API change: 9606282; both arms use this library, with A retaining
  allocating calls and the original eager projection. Supplied-buffer checks
  do not add validation overhead to the allocating control.
- Native build: SM100a, Torch 2.13, CUDA 13.0, stable torch ABI. Other runtime
  dependencies reuse the pinned stack in VALIDATION.md.

The vLLM consumer uses the existing QuantizedActivation container. Its scale
buffer follows DeepGEMM's packed, token-major scale layout. Ownership is local to
the forward/capture; no model-global temporary is introduced. Non-fused backends
retain Tensor output with explicit type guards.

## Fallback check

Initial split-KV fallback outputs were correct, but copying quantized outputs
introduced about a 3.5% TPOT regression in a spot check. The follow-up preserves
the original direct BF16 output in FULL/eager pure-decode fallback; piecewise
capture retains the quantized handoff. Correct outputs, including mixed requests,
match the baseline. The corrected fallback spot check was about 6.13 ms/token
versus baseline 6.11 ms/token, rather than the initial 6.32 ms/token. These are
regression checks, not fallback speedup measurements.

## Artifacts

artifacts/fused-output/ contains abba-{a1,b1,b2,a2}.json and logs,
abba-summary.json, capture-baseline.json, native-tests-complete.log,
fallback-baseline.json, fallback-check.json, and fallback-fixed.json.
The model driver is fused_output_bench.py; its fallback threshold parameter was
added after the A/B/B/A run. The comparison used the default always-fused route.

The final vLLM follow-up depends on #56344 and the FlashMLA API change. Full-model
ROCm and DBO are not validated for this follow-up. No GPU ownership overlaps were
observed in the measured runs; all work used the shared lease registry.

Final fallback-preserving consumer: ce578528d147b509f0d82a011c550be9f07b5d25.
A final default-route short-context rerun retained exact outputs and the
piecewise capture behavior; see final-default-check.json.
