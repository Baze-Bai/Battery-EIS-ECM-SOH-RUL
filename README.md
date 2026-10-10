# Battery EIS/ECM and cross-temperature health modeling

Research scripts for equivalent-circuit feature fitting and cross-temperature battery state-of-health modeling, including LSTM and domain-adversarial experiments.

## Source snapshot

- `EIS_model.py`: EIS processing and equivalent-circuit fitting.
- `35lstm_transfer.py`, `45lstm_transfer.py`: temperature-transfer experiments.
- `45_select_best_ad.py`: domain-adversarial experiment selection.
- `different_battery_25_lstm_transfer.py`: cross-battery experiment.
- `random_seed_experiment_*.py`: seed-based comparison scripts.

Technologies: Python, TensorFlow/Keras, SciPy, impedance.py, pandas, NumPy and scikit-learn. The related [Duke ECM dataset generator](https://github.com/Baze-Bai/baze_Dataset_project) is part of the same portfolio experience group.

## Reproduction requirements

Install `requirements.txt`. Prepare your own appropriately licensed EIS, capacity and feature files under `data/`; the scripts contain experiment-specific filenames, shapes and assumptions that require review for a new dataset. Outputs use `outputs/`. These research scripts execute experiments when run and are not a reusable production API.

Raw measurements, datasets, papers, manuscripts, model weights and experimental output are not redistributed. No publication claim or unverified improvement percentage is asserted. Static syntax checks were performed; training and numerical reproduction were not run for this export. This snapshot contains the inspected EIS/SOH experiments, not a complete runnable release of every historical RUL experiment.
