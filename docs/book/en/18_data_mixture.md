# 12. Data mixtures and deterministic sampling

Concatenating files lets file size define the training distribution. That may overrepresent web text and underrepresent code, books, or minority languages. A mixture should be specified in an effective-token budget with source weights, temperature, seed, exhaustion policy, and target tokens.

Keep a recipe table and an observed-consumption table. Filtering, source exhaustion, masks, and padding can make actual proportions differ from targets.

A distributed sampler must decide whether source selection happens globally before rank partitioning or independently on each rank. A global sample index derived from seed and step makes recovery easier to explain.

The current BatchStream samples windows from one logical token stream. It does not implement source-level mixing. The mixture schema here is a next-stage design and must not be reported as an existing feature.

Validate a mixture by generating a fixed sample plan before training, logging cumulative source tokens during training, and checking that restore continues the same sample indices.
