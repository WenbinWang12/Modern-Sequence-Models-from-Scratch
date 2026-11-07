# 16. Token Packing and Masks

Documents must become fixed-length training examples. Decide whether documents are packed across boundaries, where EOS is inserted, how padding is masked, and whether position IDs or attention masks reset at boundaries.

Pretraining usually supervises every valid next token. SFT may supervise only the response and EOS. Whether prompt tokens participate in the TTT inner loop must be part of the recipe.

Check that labels are shifted correctly, padding contributes no loss, and masks have the expected shape.
