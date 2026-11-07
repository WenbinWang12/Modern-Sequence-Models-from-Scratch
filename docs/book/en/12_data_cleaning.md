# 12. Cleaning and Quality Filtering

Cleaning controls the training distribution while preserving evidence about every decision.

```text
Read → schema validation → Unicode normalization → empty/bad text filtering
→ language and character checks → length filtering → deduplication
→ split → token statistics → manifest
```

Reports should include document and token counts before and after filtering, length quantiles, duplicate rates, and counts for every removal reason. License, PII, contamination, and safety review remain explicit stages.

Exercise: implement a deterministic cleaning command whose manifest and report are identical for the same input, configuration, and code version.
