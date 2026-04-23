# Self-Service AutoML Analytics App


This capstone project addresses that gap directly. The Self-Service AutoML Analytics App is a browser-based, no-code machine learning platform built with Streamlit and scikit-learn, designed specifically for users without a data science background. The application enables anyone to upload a dataset, select a prediction task, and immediately train, compare, and interpret multiple machine learning models through an intuitive point-and-click interface — no programming required. 

Built with [Streamlit](https://streamlit.io/) · [scikit-learn](https://scikit-learn.org/) · [XGBoost](https://xgboost.readthedocs.io/) · [SHAP](https://shap.readthedocs.io/)

---

## What It Does

The app supports three business analytics tasks:

| Task | Problem Type | Primary Metric |
|---|---|---|
| **Churn Prediction** | Binary Classification | AUC-ROC |
| **Text Classification** | Binary Classification + NLP | AUC-ROC |
| **Value Prediction** | Regression | RMSE |

After uploading your data and clicking **Run Model**, the app will:
- Auto-preprocess your dataset (type inference, encoding, imputation, datetime expansion)
- Train up to 10 models and rank them by performance
- Surface the best model's predictions at the record level
- Explain the model via SHAP plots, feature importance, or permutation importance

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/self-service-automl-analytics.git
cd self-service-automl-analytics
```

### 2. Install dependencies

Python 3.8+ is required.

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run capstone.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Requirements
streamlit
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
openpyxl
pyarrow

Install all at once:

```bash
pip install streamlit pandas numpy scikit-learn xgboost shap matplotlib openpyxl pyarrow
```
---

## Supported Data Formats

Upload any of the following file types via the sidebar:

- `.csv`
- `.xlsx` / `.xls`
- `.json`
- `.parquet`
- `.txt` (auto-detected delimiter)

---

## How to Use

1. **Upload your dataset** using the sidebar file uploader.
2. **Choose an analysis task** — Churn Prediction, Text Classification, or Value Prediction.
3. **Select your target column** (what you want to predict) and **feature columns** (your input variables).
4. *(Text Classification only)* Optionally select a **text column** to enable NLP-based signal analysis.
5. **Adjust hyperparameters** via the sidebar sliders if desired.
6. **Choose a model strategy:**
   - `Auto` — trains all available models
   - `Fast` — runs a recommended subset (Logistic/Linear Regression, Random Forest, XGBoost)
   - `Custom` — pick specific models from a list
7. Click **Run Model**.

### Output Tabs

| Tab | Contents |
|---|---|
| **Model Results** | Performance comparison chart and metrics table for all models |
| **Record Insights** | Top at-risk records with predicted scores and risk tiers |
| **Explainability & Feature Importance** | SHAP beeswarm plots (tree-based models), coefficient tables (linear models), or permutation importance (all others) |

Predictions can be downloaded as a CSV from the Record Insights tab.

---

## Models Available

**Classification (Churn Prediction / Text Classification)**

`Logistic Regression` · `Random Forest` · `XGBoost` · `SVC` · `KNN` · `Decision Tree` · `Gradient Boosting` · `AdaBoost` · `Naive Bayes` · `MLP`

**Regression (Value Prediction)**

`Linear Regression` · `Ridge` · `Lasso` · `Random Forest` · `XGBoost` · `Decision Tree` · `Gradient Boosting` · `Extra Trees` · `KNN` · `MLP`

---

## Explainability

The app applies a different explainability method depending on the best model's architecture:

| Best Model Type | Explainability Method |
|---|---|
| RandomForest, XGBoost, GradientBoosting, ExtraTrees, DecisionTree | SHAP beeswarm plot + feature importance table |
| Logistic Regression, Ridge, Lasso, Linear Regression | Signed coefficient table |
| SVC, KNN, MLP, NaiveBayes, SVR | Permutation importance |

For Text Classification tasks with a designated text column, each record additionally receives a plain-language signal summary describing the linguistic patterns that drove the classification.

---

## Text Classification — NLP Features

When a text column is provided for Text Classification tasks, the app automatically extracts:

- **Signal features** — binary flags for urgency language, financial terms, account/verification words, promotional offers, calls to action, URLs, currency symbols, uppercase ratio, and exclamation marks
- **TF-IDF features** — up to 250 weighted uni- and bi-gram term features
- **Text risk summary** — a human-readable explanation per record (e.g., *"Signals detected: urgent language, contains a link, multiple exclamation marks."*)

---

## Example Datasets

Three demonstration datasets are included in the `/data` folder:

| Task | Dataset | Source |
|---|---|---|
| Churn Prediction | `Customer_Churn.csv` | IBM Telco Customer Churn (n=7,043) |
| Text Classification | `mail_data.csv` | SMS Spam Collection — Almeida & Gómez Hidalgo, UCI MLR (n=5,574) |
| Value Prediction | `housing.csv` | Ames Housing — De Cock (2011), *Journal of Statistics Education* (n=2,930) |

---

## Project Structure

```
self-service-automl-analytics/
├── capstone.py          # Main application (single-file Streamlit app)
├── requirements.txt     # Python dependencies
├── README.md
└── data/
    ├── Customer_Churn.csv
    ├── mail_data.csv
    └── housing.csv
```
---

## References

IBM Corporation. (n.d.). Telco customer churn [Dataset]. IBM Cognos Analytics. https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn

Almeida, T. A., & Gómez Hidalgo, J. M. (2011). SMS spam collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

De Cock, D. (2011). Ames, Iowa: Alternative to the Boston housing data as an end of semester regression project. Journal of Statistics Education, 19(3), 1–15. https://doi.org/10.1080/10691898.2011.11889627

## Vanderbilt M.S. Data Science — Capstone Project (DS-5999)
Author: Kunyang Ji 

Instructor: Jesse Blocher

April 2026



