# 3. Repositories and reproducible runs

## A run is an object

A command proves that a program started. It does not explain why the result looks the way it does. A reproducible run has an identity made from source, configuration, data, environment, and state.

We write that identity as

$$
R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O),
$$

where $S$ is the source commit, $C$ the resolved configuration, $I_{\mathrm{data}}$ the fingerprints of data, tokenizer, and manifests, $E$ the environment, $K$ random and optimizer state, $Q_{\mathrm{budget}}$ the token, sequence, batch, step, and cost budget, and $O$ the outputs and evidence. Reproduction means keeping these defining objects aligned while explaining unavoidable numerical differences.

~~~text
experiment id
├── source: git commit
├── config: model / optimizer / data
├── input: manifest and SHA256
├── environment: Python / JAX / device
├── state: parameters / optimizer / sampler
└── evidence: metrics / reports / status
~~~

Without one of these fields, reproduction becomes an approximation.

## Resume semantics

resume restores parameters, optimizer state, consumed-data position, and random state. It is for continuing an interrupted run and requires compatible configuration and data.

initialize-from loads parameters but creates a new optimizer and experiment identity. It is useful when initializing TTT from the dense baseline. Confusing the two changes the meaning of step, learning rate, and effective token count.

## A small failure case

Suppose a checkpoint was written at global step 100. Its parameters have seen 100 batches, and AdamW has accumulated first- and second-moment statistics for those updates. If a new directory copies only the parameters and writes step=100, the next command may run while silently starting a different experiment:

1. the optimizer moments are zero, so the update direction changes abruptly;
2. the schedule may restart at step 0, or use a step-100 learning rate with a fresh optimizer;
3. the BatchStream random state is missing, so the next batch is different;
4. a new manifest or sequence length may change the consumed-token count while the step number still looks continuous.

“The checkpoint opens” is therefore not a recovery test. Use the same short run in two paths: one uninterrupted path and one path that saves and resumes at the middle. Compare the next input order, step, valid-token count, and loss. Low-precision reductions may differ in the last digits, but the state fields and input order must not silently change.

The opposite operation, initializing TTT from a dense baseline, creates a new identity:

~~~text
dense baseline: source=S1, data=D1, step=1000, optimizer=O_dense
TTT initialize: source=S2, data=D1, step=0, optimizer=O_ttt
~~~

The two runs share an initial parameter file but are not one continuous curve. A report must not append the second loss curve to the first.

## Code and artifacts

Even a failure during environment setup should leave a structured directory:

| File or field | Purpose |
|---|---|
| status.json | lifecycle such as running, failed, completed, or resumable |
| error.json | exception type, stage, last error, and traceback summary |
| experiment.json | commit, resolved configuration, data, and tokenizer fingerprints |
| environment.json | Python, JAX, devices, driver, and scheduler information |
| metrics.jsonl | steps, losses, valid tokens, and gradient statistics written so far |
| checkpoint/ | the latest complete state and whether resume is allowed |

This lets a reader distinguish an unavailable device, an invalid configuration, and a run that diverged after training began. An empty output directory cannot establish that a job never started; it only shows that the initialization protocol did not record the failure.

- ttt/provenance.py records Python, JAX, device, driver, PBS, and source information.
- scripts/record_environment.sh writes the environment snapshot.
- ttt/checkpoint.py saves parameters, optimizer, and sampler state and checks identity on restore.
- docs/reproducibility.md contains the complete installation and recovery path.

Machine-readable JSONL and status files are the source of truth. Markdown reports should reference them instead of manually copying numbers.

~~~bash
git rev-parse HEAD
bash scripts/record_environment.sh --output outputs/run/environment.json
sha256sum data/pretrain/manifest.json

bash scripts/run_gpu_train.sh \
  --config configs/runs/tiny_gpu.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny-gpu \
  --auto-resume
~~~

Failures should leave their output directory and error status. A failed compile or incompatible checkpoint is evidence about the boundary that still needs work.

## Transition

Two runs are comparable only when tokens, tokenizer, sequence length, effective batch, optimizer, seed, architecture, and data manifest match or are declared as variables. The next chapter makes those architectural constraints explicit through the vocabulary and configuration objects.
