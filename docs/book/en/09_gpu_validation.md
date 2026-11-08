# 9. GPU, multi-device, and checkpoint validation

## Three questions, three gates

GPU validation asks whether the code runs on the target backend, whether sharding and restore are correct, and whether the target configuration fits the memory and time budget. A tiny run answers only the first question.

The book uses three gates:

1. functional: single-device forward/backward, finite loss, causality, streaming consistency;
2. system: mesh, expert shards, all-to-all, save and restore;
3. scale: target initialization, compilation, steady throughput, peak memory, and stability.

Each gate has its own report. Failures keep their logs and environment.

## Code and commands

scripts/validate_gpu.sh runs the single-GPU architecture and streaming checks. scripts/validate_mesh.sh checks mesh, expert shards, and restore. ttt/validate.py and ttt/validate_mesh.py emit JSON reports. scripts/sutd_*.pbs requests resources; model validation remains in Python.

~~~bash
bash scripts/validate_gpu.sh outputs/gpu-validation.json
bash scripts/validate_mesh.sh \
  outputs/mesh-validation.json \
  outputs/mesh-checkpoint
~~~

Record first compilation time, steady tokens per second, peak memory, communication time, loss, gradient norms, and checkpoint size. A successful restore does not prove that the sampler state was restored.

## Failure boundaries and transition

Classify failures as device discovery, dependency or driver mismatch, XLA compile, shape or sharding, communication timeout, OOM, or checkpoint incompatibility. Keep command, commit, config, and environment for each failure.

Only after the three gates pass should the project move to 125M, 760M, long context, RULER, or official checkpoints. The next part returns to data, where larger models make provenance, cleaning, deduplication, splitting, tokenization, packing, and manifests more consequential.
