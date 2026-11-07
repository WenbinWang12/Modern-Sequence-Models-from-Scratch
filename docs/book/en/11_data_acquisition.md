# 11. Acquisition, Licensing, and Provenance

For every source, record its URL, version, license, download time, raw SHA256, and processing notes. Also record whether the data may be redistributed, whether it contains personal information, and whether it may overlap with evaluation sets.

The training corpus is not committed to Git. The repository stores schemas, processing configurations, manifests, and reports. A reproducible run must be able to identify which source versions were used even when the raw files are stored elsewhere.

Exercise: create a `source.json` record for one public corpus and attach its download command and raw checksum.
