"""
d3pm — an absorbing-state Discrete Denoising Diffusion Probabilistic Model
(D3PM) for generating synthetic malware opcode sequences.

This package is built up *incrementally* and *heavily documented* so the
implementation can be understood and verified one stage at a time:

    config.py     device detection + hyperparameter presets        (Stage 0)
    data.py       opcode loading, Vocabulary, dataset/batching      (Stage 1)
    diffusion.py  the diffusion math: schedule, q_sample, loss, sample  (Stages 2,4,6)
    denoiser.py   the transformer that predicts the clean tokens     (Stage 3)
    train.py      training loop + checkpointing                      (Stage 5)
    generate.py   sampling CLI                                        (Stage 6)
    seq_eval.py   sequence-level evaluation metrics                   (Stage 7)

Reference paper: Austin et al., "Structured Denoising Diffusion Models in
Discrete State-Spaces", NeurIPS 2021 (the D3PM paper).
"""
