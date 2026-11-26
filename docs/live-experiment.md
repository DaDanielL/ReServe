# Running the scheduler on Qwen

ReServe loads the official Qwen2.5-1.5B-Instruct weights into PyTorch. Its admission scheduler decides which waiting prompts enter the model at each iteration. Every input token goes through the model, and each output token comes from the model's logits. Wall-clock timing includes queueing, the actual forwards, scheduling, and any synchronous weight switches. Use one GPU to compare policies. [Checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)

## Local and Kaggle commands

Use an environment with Python 3.10 or newer and a PyTorch build for the intended device. The live extra pins Transformers because the runtime uses its KV-cache batching API. Commands below run in the repository root. [PyTorch installation selector](https://pytorch.org/get-started/locally/)

```bash
git clone --branch frank-scheduler https://github.com/DaDanielL/ReServe.git
cd ReServe
python -m pip install -e ".[live]"
python -m recovery_serving.live --device cuda:0 --output results/qwen-local
```

For an execution check on CPU, use the following command. A short cap makes this practical and can censor all answers. For a CPU with BF16 support, adding `--dtype bfloat16` also reduces weight memory.

```bash
python -m recovery_serving.live --device cpu --calibration 2 --requests 3 --max-new-tokens 4 --variants fixed_high recovery --output results/qwen-cpu-check
```

On Kaggle, upload the [live notebook](../notebooks/kaggle_live.ipynb), select a GPU, enable Internet for downloads, and run all cells. It uses one visible GPU even if the allocation contains two. It stores the archive in `/kaggle/working`. The CLI verifies the checkpoint and dataset SHA-256s. For offline inference, preinstall the live extra, pass the official unmodified checkpoint folder through `--model-dir`, and put the verified source files in `--data-dir` using the filenames below. [Kaggle notebooks](https://www.kaggle.com/docs/notebooks)

## What enters the experiment

| Input | Origin and how the code uses it |
| --- | --- |
| Default prompt text and reference counts | JITServe's released `traces/lmsys.json` contains 4,000 records. ReServe selects the 872 type-1 full-response records. These contain 147 distinct normalized prompts. It keeps the first row per distinct prompt, shuffles groups reproducibly and discards overlong prompts. The original input/output counts and benchmark budget travel with that row for auditing. Qwen's chat template and tokenizer determine the actual input length. [JITServe artifact](https://github.com/UIUC-MLSys/JITServe/tree/ef3ab0c3f3cf42baa6dc3f928bee7675c2efa0c6/traces) |
| Exact-checkpoint math cohort | `--source qwen-math` selects 100 MATH-500 problems and preserves all 128 published output-token counts per problem. The release identifies Qwen2.5-1.5B-Instruct, temperature 0.8, top-p 1.0 and seed 96. Of 12,800 counts, 11,983 equal 2,048, the largest recorded value. This supports a real math workload and a length reference with substantial censoring. All rollouts of one problem stay on the same side of the split. The live run generates a fresh greedy answer with the pinned checkpoint. [Dataset and revision](https://huggingface.co/datasets/kheuton/Qwen2.5-1.5B-Instruct-bon-completions/tree/d22169cef97b94649b3811bc94051495f0331829) |
| Arrival times | Contiguous eligible rows of BurstGPT's published ChatGPT conversation trace provide original timestamps and paired ChatGPT input/output counts. Rows with zero response tokens are excluded. The first selected arrival becomes time zero. `--arrival-scale 0.02` maps a 10-second gap to 0.2 seconds. Use 1 for the original spacing. ReServe attaches these arrivals in order to independently selected prompts. The manifest records the CSV line, original time and both original counts. [BurstGPT release](https://github.com/HPMLL/BurstGPT/tree/d895a53bb7b8ec137d0d2fe203b335835a78c10a) |
| Actual output length | Greedy argmax on every real forward, including the EOS token when emitted. EOS means the model ended the response. A run that hits the configured limit first is recorded as `length_cap`. Per-request output IDs make decoding and comparing answers possible. |

The cached filenames are `lmsys.json`, `math.parquet` and `BurstGPT_1.csv`. Each source choice downloads its prompt file and the BurstGPT CSV. The verified model weights occupy about 3.09 GB on disk. The run records the exact data URLs and hashes.

I found no public dataset in these papers or the checkpoint-specific search that links a Qwen2.5-1.5B-Instruct production request to its original prompt, generated output, arrival and a deadline specified by its caller. BurstGPT supplies anonymized counts. The public Qwen math release supplies prompts and sampled completions. JITServe supplies benchmark targets. Observed service latency alone cannot reveal how long the caller was willing to wait.

## How each request gets a deadline

A service-level objective (SLO) is a target for serving a request. TTFT is the time from arrival to the first output token. TPOT is the average time between subsequent output tokens. A request deadline is an absolute time measured from the start of the replayed arrival stream.

| `--deadline-mode` | Target assigned at arrival |
| --- | --- |
| `device-p95` (default) | First run separate prompts by themselves with the actual checkpoint in high precision. Estimate the new request's output length using the 80th percentile of observed calibration counts in its power-of-two input-length bucket, falling back to all calibration requests for an empty bucket. For each timing observation, compute its TTFT plus its TPOT times the predicted number of tokens after the first. Take the empirical 95th percentile of these estimated times and add it to arrival. The timing pool uses the same input-length bucket when at least five observations are available, otherwise the whole cohort. A one-token answer contributes its measured TTFT. Extrapolation uses the median observed TPOT or, for an entirely one-token timing cohort, a measured decode step from warmup. |
| `jitserve-20s` | Arrival plus 20 seconds. JITServe §6.1 obtained this full-response benchmark target from the 95th-percentile latency of 1,000 DeepSeek API calls. Useful as a fixed external reference across devices. |
| `artifact` | JITServe type-1 records only. Interpret the stored `deadline` column as a relative benchmark budget in milliseconds, divide by 1,000 and add to arrival. For example, source row `lmsys:27` stores 2,701 ms. The paper's benchmark client also supports overwriting these labels with configured type- and collection-based targets. |
| `provided` | Read a CSV via `--budgets path/to/budgets.csv` with `source_id,prompt_group,budget_s`. Use externally recorded client budgets if available. The same format is exported by each run so that both devices can be scored against identical frozen targets. The caller supplies the provenance for their own budgets. |

The device rule is our adaptation of JITServe's measurement-based target setting. Its P95 comes from timings on your machine. It estimates a feasible service budget for a bounded workload. It does not estimate an individual user's preference or promise 95% attainment under load. Calibration and evaluation have disjoint prompt hashes. Evaluation response lengths, source reference answers, eventual EOS, and offline burst labels are excluded from deadline estimation and online admission. All policies and repetitions receive the same absolute deadlines. The artifact mode deliberately uses the paper's labels. [JITServe §6.1](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf) · [JITServe client](https://github.com/UIUC-MLSys/JITServe/blob/ef3ab0c3f3cf42baa6dc3f928bee7675c2efa0c6/benchmark/trace/client.py)

The historical CPU-only BurstGPT replay now uses DistServe's OPT-13B chatbot streaming targets of 0.25 seconds to first token and 0.10 seconds per subsequent token. Its derived completion target is arrival plus 0.25 seconds plus 0.10 seconds times the predicted output count after the first. DistServe evaluated streaming objectives separately. This conversion to an end-to-end deadline is specific to ReServe. The predictor uses measured lengths from a separate BurstGPT window. Old synthetic fixtures remain available for regression testing. [DistServe Table 1 and §6.1](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)

## What the scheduler controls

- All six variants use the same model, dataset split, arrival times, cache accounting, and measurement code. Fixed high and fixed low establish reference points. The dynamic variants use FIFO, estimated deadline slack, a drain rule, or the recovery forecast. Repeats shuffle policy order using the recorded seed.
- The scheduler admits requests at token iteration boundaries. Every active request gets a step. Prompts of equal length can share a prefill forward. Active requests with equal cache lengths can share a decode forward. Other groups execute sequentially. Prefill processes the prompt and builds the initial key/value (KV) cache. The KV cache holds prior attention keys and values for reuse during generation. [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)
- High precision uses FP16 by default on CUDA. Low precision packs the weights of the three feed-forward projections in the final two Qwen layers into four bits per value. All other weights and the KV dtype stay at the high setting. The implementation retains high and packed masters in host memory, copies the active representation to the device on a switch, and unpacks tiles for computation. Requests preserve their existing KV across a switch. Timestamps mark when the transfer and device synchronization have finished. This portable reference implementation has its own transfer and kernel overhead. MorphServe uses optimized asynchronous execution and a layer sensitivity policy. [MorphServe](https://proceedings.mlsys.org/paper_files/paper/2026/hash/8144a9d62e506af0fcdeac0e456b2710-Abstract-Conference.html)
- The low-mode admission ceiling adds the actual difference in stored weight bytes divided by Qwen's KV bytes per token to the specified high-mode headroom. Accounting uses 16-token blocks. Hugging Face allocates real dense cache tensors. The block count is a logical scheduler budget and the CUDA log reports actual allocated, reserved and peak bytes. This setup can exercise constrained headroom on a single GPU. It does not establish fragmentation, saturation of the entire device or deployment-kernel throughput. [PagedAttention](https://arxiv.org/abs/2309.06180)
- Under low-mode recovery pressure, the scheduler estimates which running requests will finish within a short horizon and release their KV. The experiment tests whether using that forecast for admission lets high weights return earlier at a given deadline goodput. Output-length prediction and lookahead admission have prior art. The comparison of their effect on **precision restoration** is the project's contribution under test. [Spexis §5](https://aclanthology.org/2026.emnlp-main.29/) · [Past-Future Scheduler](https://arxiv.org/abs/2507.10150)

## Reading a result

Start with `run.json` (hardware, versions, inputs, settings and calibration cap count), `workload.json` (every offered arrival) and `summary.json` (all variants). `calibration-high.json` and `calibration-low.json` hold isolated Qwen timings and lengths. Each variant saves request-level observations and an iteration and switch event log. Initialization, downloads and warmup precede measurement. Policy time includes queue and scheduler work and synchronous instrumentation. Measured inference on CPU can verify correctness. CUDA results require a CUDA run.

| Measure | Interpretation |
| --- | --- |
| Deadline attainment | Naturally ended, on-time requests divided by all offered requests. A capped response contributes zero. |
| Deadline goodput | On-time, naturally ended requests per second from the first arrival to the last frozen deadline. This denominator is shared by all policies. The model drains the queue to observe late completions. |
| Throughput | All generated tokens, including capped responses, divided by measured elapsed time through drain and final restoration. |
| Latency and fairness | P95 queue time, TTFT and mean per-request TPOT. End-to-end latency is reported for natural completions. The long/short groups use the median source output count and a declared cutoff so that a policy cannot change its group membership by altering the answer. |
| Recovery | The measured return to high precision relative to the end of the offline peak-arrival interval. Fixed-mode, no-switch and censored episodes have distinct labels. The event stream reports actual activation, restore intent and successful synchronization. |
| Prediction error | Mean absolute difference in tokens between the frozen prediction and the actual generated length for natural completions. Check its sample count and the cap fraction before interpreting it. |

For a reportable comparison, keep the same device model, precision, data hashes, seed, cap and budget file. Inspect the capped fractions in calibration and evaluation. Increase the generation cap and high-mode headroom together if long responses are cut short, then recalibrate. The mathematical cohort in particular can require longer runs. Repeating with `--repeats 3` gives separate wall-clock observations per policy. The measured experiment targets scheduler behavior under these settings. Any claim about answer quality requires a separate quality evaluation.

## Verification on CPU

The pinned 1.5B checkpoint ran with PyTorch 2.6.0+cpu, BF16, and Transformers 4.51.3. An execution check ran all six variants on three JITServe prompts, with one swappable layer and a four-token cap. All three answers reached the cap in each variant. Each dynamic variant switched to packed weights and returned to high precision. A separate fixed-high check loaded the exact-checkpoint math dataset and generated from its prompt.

For JITServe row `lmsys:253`, the artifact reports 57 input tokens and a one-second benchmark budget. Qwen's chat template produced 85 input tokens. The CPU run generated token 2753 followed by EOS 151645 and missed the one-second budget. This single request checks token accounting, natural termination and deadline scoring. The Kaggle notebook's schema and cells were validated. A Kaggle GPU measurement remains outstanding.
