# 6. Cleaning: every filter changes the distribution

## Clean is not a natural label

A navigation template, a short code function, and a mathematical proof may all contain punctuation or few alphabetic characters. A rule designed for web prose can delete useful code or mathematics.

Cleaning should reduce obvious noise while measuring false deletions. Treat filters as experiment variables: document their order, thresholds, source-specific retention, and sampled decisions.

## Three kinds of processing

Separate format repair, rule-based rejection, and content review. Unicode normalization, empty-text checks, length limits, language detection, PII review, and safety review have different meanings and different failure modes.

A stable rejection taxonomy might include invalid_schema, empty, too_short, too_long, bad_unicode, language_mismatch, and safety_review. Keep the first rejection reason separate from the full set of triggered diagnostics so counts can be reconciled.

For input documents, a report should satisfy

$$
n_{\rm input}=n_{\rm kept}+\sum_r n_{\rm rejected,r}.
$$

The denominator must be explicit: retention after schema validation is different from retention relative to the raw source.

## Auditing

Sample retained, rejected, near-threshold, and rare-source documents with a fixed seed. Inspect length quantiles and source/language retention; an unchanged overall retention rate can hide the loss of a small but important domain.

The current ttt.data implementation performs limited field and whitespace checks. It does not provide a complete quality, language, PII, or safety pipeline. This chapter specifies the upstream contract rather than claiming those filters are already integrated.

The next chapter removes exact and near duplicates using the same normalization version.
