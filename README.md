# Sensitivity Fidelity Reveals Physical Dependencies Beyond Forecast Accuracy in Deep-Learning Earth-System Models

This directory contains the code for the Rayleigh–Bénard (RB) experiments in the paper **“Sensitivity Fidelity Reveals Physical Dependencies Beyond Forecast Accuracy in Deep-Learning Earth-System Models.”**

The code includes training programs for four RB models: IFactFormer, FNO, AFNO, and Swin. Model implementations are located in `models/`, RB data loading and training utilities are located in `libs/`, and `sensitivity.py` is the entry point for sensitivity calculations.

References for the RB data simulation code and adjoint experiments:

- [DedalusProject/dedalus](https://github.com/DedalusProject/dedalus)
- [csskene/dedalus_adjoint_examples](https://github.com/csskene/dedalus_adjoint_examples)

Reference for the ENSO experiments:

- [QiChen-7-c/CTEFNet](https://github.com/QiChen-7-c/CTEFNet)
