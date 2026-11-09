# 3. Repositories and reproducible runs

## A run is an object

A command proves that a program started. It does not explain why the result looks the way it does. A reproducible run has an identity made from source, configuration, data, environment, and state.

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

## Code and artifacts

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
