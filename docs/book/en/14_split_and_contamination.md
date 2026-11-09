# 7. Splits and contamination

[中文版](../14_split_and_contamination.md)

Randomly assigning records is not enough when mirrors, versions, authors, or repositories are related. Choose the split unit for the evaluation question: document cluster, project, time window, or another stable group.

A deterministic hash split can assign a stable group to train, validation, or test:

~~~python
import hashlib

def choose_split(group_id):
    value = int.from_bytes(
        hashlib.sha256(group_id.encode()).digest()[:8],
        "big",
    ) % 10_000
    if value < 100:
        return "test"
    if value < 300:
        return "validation"
    return "train"
~~~

This controls group assignment, not exact token proportions. Long groups can dominate token counts. Split before tokenization and packing so random windows cannot overlap across sets.

Contamination checks compare training input with public evaluation data using exact, long-substring, or approximate matches. Report the method, candidate count, reviewed count, and unresolved cases. A zero exact match does not prove that related content is absent.

The next chapter fixes the tokenizer and vocabulary that turn each split into IDs.
