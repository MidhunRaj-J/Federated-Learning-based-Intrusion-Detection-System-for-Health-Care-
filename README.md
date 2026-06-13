# Federated Learning-Based Intrusion Detection System for Health Care

Hybrid federated intrusion detection research notebook that combines an ANN and LightGBM ensemble for multi-class attack detection on the CICIoMT dataset.

The repository currently centers on a single Jupyter notebook, [model (1).ipynb](model%20%281%29.ipynb), which documents the approach, shows preview visualizations, and contains the full training pipeline for the federated workflow.

## Overview

This project implements a hybrid federated learning pipeline with:

- Federated averaging for ANN weight aggregation.
- Local LightGBM training on each client.
- Dynamic ensemble weighting between ANN and LightGBM predictions.
- Class balancing, early stopping, dropout, batch normalization, and learning-rate reduction to improve stability.

The notebook is written as a research-style implementation and includes architecture notes, methodology, preview graphs, dataset statistics, and the end-to-end training code.

## Features

- Federated learning simulation with multiple clients and multiple rounds.
- ANN model with BatchNormalization and Dropout layers.
- LightGBM model trained per client.
- Dynamic ensemble scoring using validation accuracy.
- Local model evaluation for each client.
- Confusion matrix, classification report, and multi-plot result dashboard.
- Saved artifacts such as model weights, metrics, scalers, label encoders, and training logs.

## Notebook Contents

The notebook includes:

1. Project introduction and architecture overview.
2. Mermaid and text-based diagrams for the federated workflow.
3. Preview visualizations with sample data.
4. Data flow and training-process documentation.
5. Dataset statistics preview.
6. Package installation cell.
7. Main federated training pipeline.
8. Results, metrics, and usage notes.

## Requirements

The main notebook uses the following Python packages:

- numpy
- pandas
- scikit-learn
- lightgbm
- joblib
- tensorflow
- matplotlib
- seaborn
- tqdm
- kagglehub

## Setup

Use a Python environment with Jupyter support, then install the dependencies:

```bash
pip install numpy pandas scikit-learn lightgbm joblib kagglehub tqdm matplotlib seaborn tensorflow
```

If you want to rely on KaggleHub dataset loading, make sure the KaggleHub package can access the dataset referenced in the notebook.

## How to Run

1. Open [model (1).ipynb](model%20%281%29.ipynb) in Jupyter or VS Code.
2. Run the preview cells if you want to inspect the expected output format.
3. Run the installation cell if dependencies are not already installed.
4. Execute the main training cell.

For a faster test run, pass a smaller sample fraction near the bottom of the training cell, for example:

```python
metrics, local_results, history = main(
    num_clients=3,
    global_rounds=2,
    local_epochs=3,
    local_batch=256,
    clients_per_round=2,
    sample=0.1
)
```

## Data Expectations

The notebook expects preprocessed CICIoMT parquet files or a KaggleHub-accessible equivalent:

- `training_preprocessed.parquet`
- `testing_preprocessed.parquet`

If those files are not available locally, the notebook attempts to load the dataset through KaggleHub.

## Outputs

Running the main training pipeline creates an output folder named `fidchain_out_hybrid_final/` containing:

- `global_ann_final.keras`
- `final_lgbm_models.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
- `metrics.json`
- `ledger.json`
- `summary_metrics.csv`
- `local_model_results.csv`
- `classification_report.txt`
- `comprehensive_results.png`

Intermediate round checkpoints and client-specific LightGBM models may also be saved during training.

## Notes

- The notebook is currently research-oriented and not packaged as a standalone Python module.
- The preview charts use dummy data; real plots are produced only after running the full training pipeline.
- The current repository does not include the dataset files, so you will need to supply them or use the KaggleHub path configured in the notebook.

## Repository Structure

```text
.
├── LICENSE
├── README.md
└── model (1).ipynb
```

## License

See [LICENSE](LICENSE) for licensing details.