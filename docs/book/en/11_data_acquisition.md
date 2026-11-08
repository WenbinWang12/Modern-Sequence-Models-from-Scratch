# 11. Data acquisition, licensing, and provenance

## Before data enters the repository

Data acquisition is a source-management problem before it is a download command. We need to know who published the data, which version was obtained, whether research and redistribution are allowed, and whether the same snapshot can be retrieved later.

A source record should include the URL, release or snapshot version, download time, license, raw-file SHA256, and notes about restrictions. The repository stores schemas, source records, processing configuration, manifests, and reports; large raw files live in controlled storage.

## Review boundaries

Check the raw license, tokenizer training data, embedded third-party content, personal information, benchmark contamination, and the planned model-release terms separately. Publicly downloadable does not mean freely redistributable.

## Artifacts and transition

Each processing stage records input digest, configuration, code commit, output digest, document and token counts, and filtering counts. The next chapter turns this provenance into explicit cleaning rules.
