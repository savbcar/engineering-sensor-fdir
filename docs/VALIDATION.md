# Validation Strategy

The repository uses four levels of validation:

1. **Unit tests** verify deterministic simulation, fault injection, FFT frequency recovery, anti-alias helper behavior, filter attenuation, feature extraction, detector execution and metric calculations.
2. **End-to-end execution** runs the default 240 s / 250 Hz scenario, writes all data products, trains both unsupervised models, serializes models and generates figures.
3. **Ground-truth metric checks** compare fault-specific residual flags against exact injected sample labels and compare all detector families on a common overlapping-window grid.
4. **Repository checks** verify importability, CLI execution, Python byte-code compilation and clean Git-ready structure. Ruff/mypy and Docker are defined in the project and CI; whether they are locally executable depends on tool availability.

A passing synthetic validation run is necessary but not sufficient for real-world reliability. The next evidence tier should use independent experimental or public engineering sensor datasets.
