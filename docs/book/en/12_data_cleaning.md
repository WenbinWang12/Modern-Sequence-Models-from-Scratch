# 12. Data cleaning and quality filters

## What cleaning controls

Cleaning should control obvious noise while preserving an explainable distribution. Over-filtering removes code, long documents, or minority languages; under-filtering spends tokens on templates, corrupted text, navigation pages, and boilerplate.

~~~text
read
→ schema validation
→ Unicode normalization
→ empty and malformed text filters
→ language and character checks
→ length filters
→ PII and safety review
→ clean documents
~~~

Every filter has a stable reason such as empty, invalid_schema, too_long, bad_unicode, or language_mismatch.

## Statistics

The report records input and output documents, characters and bytes, length quantiles, counts by reason, retention by source and language, processing time, configuration, and commit. The same input, configuration, and code version must produce the same order, report, and digest.

Cleaning does not solve licensing, PII, or evaluation contamination. Those remain explicit review stages. The next chapter handles exact and near duplicates.
