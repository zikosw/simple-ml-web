# Simple ML Web

A single-page, fully client-side machine learning web app to **train and validate models** in the browser — no server, no build step, and no data ever leaves your machine.

Implemented with [ml.js](https://github.com/mljs/ml) (bundled locally in `vendor/ml.min.js`).

## Features

- **Two classifiers**: Gaussian Naive Bayes and Random Forest
- **Datasets**: built-in Iris sample dataset, or upload your own CSV
  - CSV format: header row, numeric feature columns, last column is the class label
- **Training options**: adjustable test split, number of random forest trees, optional shuffling
- **Validation**: accuracy and confusion matrix for each model on a held-out test set
- **Prediction**: classify new samples with both trained models

## Usage

Open `index.html` in a browser, or serve the folder with any static server:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

1. Load the Iris sample dataset or upload a CSV
2. Choose training options and click **Train & validate models**
3. Compare accuracy and confusion matrices
4. Enter feature values and click **Predict**
