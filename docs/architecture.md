# Implementation boundaries

## Starter path

`YAML -> seeded arrival trace -> client admission queue -> mock backend -> logs`

This path works on a CPU. The queue limits concurrent client calls. It does not
perform continuous batching, schedule individual decoding steps, or manage KV memory.
`VLLMBackend` and `RecoveryAwareScheduler` deliberately raise `NotImplementedError`.
`PrecisionController` is a protocol, not a functioning controller.

## Real system path to build

1. Choose and pin an inference engine, model revision, GPU, and quantization backend.
2. Implement fixed-precision trace replay and obtain real engine-side metrics.
3. Add per-request and per-allocation identifiers so KV ownership is observable.
4. Implement safe precision transitions and elastic cache capacity, retaining active state.
5. Profile recovery using the existing baseline policy.
6. Add the recovery-aware policy only after identifying avoidable waiting.

A streaming HTTP client can benchmark an engine, but controlling request order at
the HTTP layer does not provide token-level scheduling or KV placement. Those
parts likely require a versioned engine fork and allocator changes. Linking a
fork's exact commit and installation procedure is preferable to undocumented patches
in a notebook. Low-level CUDA/VMM work is an implementation choice to assess, not
a working feature of this template.

## Events to instrument in the real implementation

Use a monotonic clock for intervals and correlate CPU timestamps with CUDA events.
Avoid synchronizing the GPU after every token just to log timings.

| Event | Required information |
| --- | --- |
| request arrival / admission / finish | request ID, planned/actual time, deadline |
| first token / subsequent output | request ID, timestamp, token count |
| KV allocation / free | request ID, allocation unit, bytes, memory region |
| low precision requested / active | requested mode, layers, request/activation time |
| recovery requested | precise policy trigger and timestamp |
| unit becomes reclaimable | unit, bytes, last owner, GPU-safe time |
| restore transfer submitted / completed | unit/layer, bytes, stream, timestamps |
| high precision active | actual execution mode and completion timestamp |

Logical block release may precede completion of in-flight GPU reads. A unit must
be GPU-safe before its data is overwritten. Distinguish this from a client request
being marked complete.

Track tensor-allocated memory, allocator-reserved memory, actual active KV bytes,
and fully reclaimable allocation units separately. Device memory usage alone
cannot identify the reason precision restoration is delayed.

The higher-precision mode resuming does not undo previously generated tokens or
recompute all cached states. Measure actual answer quality separately from time
spent in each precision mode.
