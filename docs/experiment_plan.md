# Experiment plan

## Milestones

- [ ] Run the supplied CPU smoke check locally and in Kaggle.
- [ ] Select a small model and compatible fixed-precision engine on one GPU.
- [ ] Run a baseline burst and log queueing, TTFT, decoding latency, and KV occupancy.
- [ ] Add and verify real precision switching without dropping active request state.
- [ ] Separate recovery waiting from restore transfers and their overlap.
- [ ] Implement a recovery-aware policy and compare against existing recovery behavior.
- [ ] Evaluate quality, deadline goodput, tail latency, and long-request waiting.

## Initial baseline matrix

| Weights | Scheduling / recovery |
| --- | --- |
| Fixed higher precision | Standard engine scheduler |
| Fixed lower precision | Same engine scheduler |
| Dynamic precision | Existing pressure-aware recovery policy |
| Dynamic precision | Same mechanism plus independent cost-aware scheduling |
| Dynamic precision | Proposed cost-and-recovery-aware policy |
| Dynamic precision | Simple temporary-region draining policy |

Use the same engine and memory budget where possible. A mock run is never a row
in this matrix. A reload that discards active cache is not equivalent to a
state-preserving precision transition; label such a baseline explicitly.

## Controlled variables

Vary burst intensity, duration, spacing, and input/output length mixture. Save the
trace once and replay it across policies. Repeat experiments over several seeds.
Use real prompts for quality evaluation; synthetic token counts only exercise load.
Keep warmup, prefix caching, kernel choices, and device configuration consistent.

## Recovery definition

Define burst end from the arrival trace and define the recovery trigger separately.
Report both burst-end-to-high-precision time and trigger-to-high-precision time.
Choose whether completion means all layers or a stated target fraction restored.
Mark runs that never recover within the observation window as censored/unrecovered,
rather than dropping them or reporting a zero recovery time.

Instrument the critical path: the wait for eligible memory and transfer work can
overlap, so do not assume total recovery is their simple sum. Compare with an
oracle remaining-length policy only as an upper bound; practical policies use
predictions and should be evaluated under prediction error.

## Required reporting

- Deadline goodput and the precise deadline/SLO definition.
- P95/P99 TTFT, inter-token gaps, and end-to-end latency.
- Recovery time, switching overhead, memory occupancy, and reclaimable units.
- Time/tokens at each precision, plus independent output-quality scores.
- Submitted, completed, failed, rejected, timed-out, and unrecovered counts.
- Long-request waiting and deadline misses by length group.
- Policy CPU overhead and repeat-run variability.

Faster recovery obtained just by rejecting work or indefinitely delaying long
requests does not establish a better policy. If transfers dominate the critical
path, report that finding even if the scheduler provides little improvement.
