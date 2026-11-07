# 18. Mixtures and Deterministic Sampling

Multi-source corpora should be sampled by an explicit recipe rather than concatenated implicitly.

```json
{
  "sources": [
    {"name":"web", "manifest":"data/web/manifest.json", "weight":0.5},
    {"name":"books", "manifest":"data/books/manifest.json", "weight":0.3},
    {"name":"code", "manifest":"data/code/manifest.json", "weight":0.2}
  ],
  "temperature": 1.0,
  "seed": 1234
}
```

Specify how source weights, temperature, global steps, and resume interact. A fixed seed must reproduce the same source sequence after recovery.
