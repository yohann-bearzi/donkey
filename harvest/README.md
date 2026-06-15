# Stage 1 -- Harvest

Run the trunk over text corpora and capture, per emitted token position: the pre-norm hidden state (the drafter's prediction target), the emitted token id, and the trunk's top-p token distribution (the distributional target).

Output is memmapped binary under traces/<corpus>/ (see ARCHITECTURE.md for the format). The dataset is large (hundreds of GB of hidden states) and read on demand during training, not held in memory.

- harvest.py : the capture driver
- verify_hiddens.py : sanity checks on a harvested corpus (shapes, alignment, coverage)

Known gap (WIP): prompt tokens are not currently captured, only emitted tokens. This conflates novel and prompt-copied tokens in the analysis -- closing it is planned and unblocks the largest bucket of rare-token misses (see FINDINGS.md).
