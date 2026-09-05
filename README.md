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

## Deploy (GitHub Pages)

This repo now includes a GitHub Actions workflow at `/home/runner/work/simple-ml-web/simple-ml-web/.github/workflows/deploy-pages.yml` that deploys the static site to GitHub Pages on every push to `main`.

1. In GitHub, open **Settings → Pages**
2. Under **Build and deployment**, set **Source** to **GitHub Actions**
3. Push to `main` (or run the **Deploy static site to GitHub Pages** workflow manually)
4. Access the site at:

`https://zikosw.github.io/simple-ml-web/`
