# 19. Data Preflight

Data errors should fail before expensive training starts.

```text
schema → manifest → SHA256 → vocabulary range → mask shape
→ split overlap → tokenizer/config match → sequence length
→ batch and parallelism divisibility
```

Write a `data-preflight.json` report beside the run configuration. It becomes part of the checkpoint context and makes a failed run diagnosable.
