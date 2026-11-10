# 21. GPU, multi-device, and checkpoint validation

## Three questions, three gates

A tiny GPU run answers only “can this graph execute on this backend?” It does not prove that sharding is correct, that an expert communication path restores correctly, or that the target model fits the memory and time budget. Treat validation as three gates with separate reports:

| Gate | Question | Evidence |
|---|---|---|
| Functional | Is the single-device computation causal and numerically finite? | finite loss/gradients, causality, full/streaming logits |
| System | Are mesh, shards, communication, and restore correct? | global/local shapes, all-to-all, next-step continuity |
| Scale | Does the target configuration fit? | compile time, steady throughput, peak memory, stability |

A pass at one gate cannot hide a failure at another. Keep the command, commit, configuration, environment, and logs for every failure.

## Gate 1: functional semantics

`scripts/validate_gpu.sh` runs architecture, causality, gradient, and streaming checks. It calls `ttt.validate`, which builds tiny dense, GDN, MoE, Loop, and context-length cases. The validator checks that loss and gradients are finite, that changing future tokens does not change earlier logits, and that training and streaming paths agree within the reported tolerance.

For TTT, the functional report should also show that an enabled inner update changes fast weights, a disabled update leaves the control path stable, padding in a partial chunk does not adapt the state, and a session snapshot restores the same next logits. These tests use small shapes but must include the actual layer types, cache, and adaptation target used by the configuration.

~~~bash
bash scripts/validate_gpu.sh outputs/gpu-validation.json
~~~

The script records an environment report before running Python. Its JSON includes finite-loss status, logits error, state bytes, compilation time, and execution time. It is a semantic gate, not a language-quality result.

## Gate 2: mesh and restore

JAX mesh is a placement description, not merely “copy the array to every GPU.” A data axis partitions or replicates batches; an expert axis places expert parameters and moves routed tokens. Their combination changes communication and checkpoint sharding.

The mesh report should include the global shape and each rank's local shard shape, process and device identifiers, parameter and optimizer sharding, route token counts, all-to-all time, and the rank that failed if a collective times out. Printing only rank-0 loss can hide an error on another rank.

~~~bash
bash scripts/validate_mesh.sh \
  outputs/mesh-validation.json \
  outputs/mesh-checkpoint
~~~

`ttt/validate_mesh.py` exercises expert parallel initialization, two validation steps, checkpoint save, and restore. The test is a systems gate: it proves the current mesh path and artifact schema, not that a large run will fit.

## Gate 3: target scale and timing

JAX's first execution usually includes tracing and XLA compilation. Split wall-clock time into

$$
T_{\mathrm{job}}=T_{\mathrm{setup}}+T_{\mathrm{compile}}+T_{\mathrm{steady}}+T_{\mathrm{eval}}+T_{\mathrm{checkpoint}}.
$$

Steady throughput is valid tokens divided by \(T_{\mathrm{steady}}\), after warm-up. Including compilation, evaluation, or checkpoint I/O in that denominator reports the scheduler and filesystem as if they were model kernels. Record the measurement window, asynchronous synchronization point, dtype, preallocation setting, and peak-memory interface.

Scale validation also needs failure boundaries: device discovery, driver or dependency mismatch, XLA compilation, shape or sharding, collective timeout, OOM, and checkpoint incompatibility are different problems and should remain distinct labels.

## Scheduler and model layers

The `scripts/sutd_*.pbs` files request resources, set process information, and choose shared directories. `ttt/validate.py` and `ttt/validate_mesh.py` own model semantics. Moving to another cluster changes scheduler fields, coordinator address, process count, process ID, and filesystem paths; it should not change what “causal consistency” or “restore success” means. These values must come from the allocation rather than being hard-coded in a Python module.

## Checkpoint restore boundaries

Restore validation should cover:

1. normal resume on the same device and mesh;
2. process restart after a job interruption;
3. explicit rejection when batch or data identity changes;
4. explicit conversion or rejection when device layout changes;
5. separation between training checkpoints and serving-session snapshots;
6. equality or documented tolerance of the next step's state and metrics.

Opening a file is not enough. A successful restore means the next state transition is the one the saved run specifies.

Only after the functional and system gates pass should the project submit long target-scale jobs. A quality table may include a variant only when its scale report records compile time, steady throughput, memory, checkpoint behavior, and failures. Official checkpoints, long context, and RULER require additional data and hardware; tiny results should be labeled as hypotheses or partial reproductions until those conditions are met.
