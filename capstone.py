# Locate the file: 'cd ~/Desktop/Capstone'
# Execute locally with: 'streamlit run capstone.py'

import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


FRAUD_SIGNAL_PATTERNS = {
    "urgency_terms": r"\b(urgent|immediately|now|asap|limited)\b",
    "financial_terms": r"\b(payment|bank|credit|card|invoice|refund|wire|cash)\b",
    "account_terms": r"\b(account|password|verify|login|security|otp|code)\b",
    "offer_terms": r"\b(free|winner|won|offer|gift|bonus|reward)\b",
    "action_terms": r"\b(click|tap|reply|call|download|open|visit)\b",
}

FRAUD_POSITIVE_LABEL_HINTS = [
    "fraud",
    "spam",
    "scam",
    "suspicious",
    "risk",
    "positive",
    "yes",
    "true",
    "1",
]


st.set_page_config(page_title="AutoML Customer Analytics App", layout="wide")

# ── Print / export CSS ──────────────────────────────────────────────────────
# When the user prints or saves as PDF (Ctrl/Cmd+P), the sidebar is hidden
# and all charts expand to full page width for a clean single-column layout.
st.markdown(
    """
    <style>
    @media print {
        /* Hide sidebar and its toggle button */
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        button[kind="header"] {
            display: none !important;
        }
        /* Expand main content to full width */
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        /* Make matplotlib/plotly figures scale to full width */
        [data-testid="stImage"] img,
        [data-testid="stPyplotRootElement"] > div {
            width: 100% !important;
            max-width: 100% !important;
        }
        /* Avoid page breaks inside charts and dataframes */
        [data-testid="stDataFrame"],
        [data-testid="stPyplotRootElement"] {
            page-break-inside: avoid;
        }
        /* Hide the action toolbar (download/fullscreen icons on dataframes) */
        [data-testid="stElementToolbar"] {
            display: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AutoML Customer Analytics App")
st.caption(
    "Choose an analysis task, upload a dataset, and compare multiple models with optional text analysis."
)


@st.cache_data
def load_data(file):
    file_extension = file.name.split(".")[-1].lower()
    try:
        if file_extension == "csv":
            return pd.read_csv(file)
        if file_extension in ["xlsx", "xls"]:
            return pd.read_excel(file)
        if file_extension == "json":
            return pd.read_json(file)
        if file_extension == "parquet":
            return pd.read_parquet(file)
        if file_extension == "txt":
            return pd.read_csv(file, sep=None, engine="python")
    except Exception as exc:
        st.error(f"Error loading file: {exc}")
    return None


def estimate_runtime(n_rows, n_features, selected_list, has_text_features=False):
    # Empirical, intentionally conservative estimate for interactive demo usage.
    row_factor = (max(n_rows, 1) / 5000) ** 0.65
    feature_factor = (max(n_features, 1) / 20) ** 0.45
    base = max(0.08, row_factor * feature_factor)

    model_weights = {
        "Logistic": 0.35,
        "LinearRegression": 0.30,
        "Ridge": 0.35,
        "Lasso": 0.45,
        "RandomForest": 1.00,
        "XGBoost": 1.10,
        "SVC": 1.40,
        "KNN": 0.60,
        "DecisionTree": 0.45,
        "GradientBoosting": 0.90,
        "AdaBoost": 0.70,
        "NaiveBayes": 0.25,
        "ExtraTrees": 0.95,
        "SVR": 1.20,
        "MLP": 1.20,
    }

    total = 0
    for model_name in selected_list:
        total += base * model_weights.get(model_name, 0.70)

    if has_text_features:
        total += max(0.15, min(1.25, n_rows / 12000))

    return total


def build_regression_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=5000),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "ExtraTrees": ExtraTreesRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }


def get_models(problem_type, model_mode, selected_models, params):
    if problem_type == "Classification":
        all_models = {
            "Logistic": LogisticRegression(
                C=params["C_value"],
                max_iter=1000,
                class_weight="balanced",
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=params["rf_n_estimators"],
                max_depth=params["rf_max_depth"],
                random_state=params["random_state"],
            ),
            "XGBoost": XGBClassifier(
                n_estimators=params["xgb_n_estimators"],
                max_depth=params["xgb_max_depth"],
                learning_rate=params["xgb_learning_rate"],
                eval_metric="logloss",
                random_state=params["random_state"],
            ),
            "SVC": SVC(
                probability=True,
                class_weight="balanced",
                cache_size=700,
                random_state=params["random_state"],
            ),
            "KNN": KNeighborsClassifier(n_neighbors=params["knn_neighbors"]),
            "DecisionTree": DecisionTreeClassifier(
                max_depth=params["rf_max_depth"],
                random_state=params["random_state"],
            ),
            "GradientBoosting": GradientBoostingClassifier(
                random_state=params["random_state"]
            ),
            "AdaBoost": AdaBoostClassifier(random_state=params["random_state"]),
            "NaiveBayes": GaussianNB(),
            "MLP": MLPClassifier(max_iter=300, random_state=params["random_state"]),
        }
        fast_models = ["Logistic", "RandomForest", "XGBoost"]
    else:
        all_models = build_regression_models()
        fast_models = ["LinearRegression", "RandomForest", "XGBoost"]

    if model_mode == "Auto (Run all models)":
        chosen = list(all_models.keys())
    elif model_mode == "Fast (Recommended set)":
        chosen = fast_models
    else:
        chosen = selected_models or fast_models

    return {name: all_models[name] for name in chosen if name in all_models}


def clean_feature_names(columns):
    return [re.sub(r"[\[\]<>\s,]", "_", str(col)) for col in columns]


def make_safe_term_name(term):
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(term).strip().lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "term"


def prettify_feature_name(feature_name):
    label = str(feature_name)
    if label.startswith("text_term_"):
        return label.replace("text_term_", "").replace("_", " ")
    if label.startswith("text_signal_"):
        return label.replace("text_signal_", "").replace("_", " ")
    generic_prefixes = ["message_", "message ", "text_", "text ", "description_", "description "]
    lower_label = label.lower()
    for prefix in generic_prefixes:
        if lower_label.startswith(prefix):
            label = label[len(prefix):]
            break
    return label.replace("_", " ")


def encode_binary_target(y, task_mode=None):
    y_series = y.copy()
    if pd.api.types.is_numeric_dtype(y_series):
        y_numeric = pd.to_numeric(y_series, errors="coerce")
        unique_values = sorted(v for v in y_numeric.dropna().unique())
        if len(unique_values) == 2:
            return pd.Series(y_numeric, index=y_series.index, name=y_series.name), {
                0: unique_values[0],
                1: unique_values[1],
            }
        return pd.Series(y_numeric, index=y_series.index, name=y_series.name), None

    normalized = y_series.astype(str).str.strip()
    unique_labels = [label for label in pd.unique(normalized) if label != "nan"]

    if len(unique_labels) != 2:
        encoder = LabelEncoder()
        encoded = pd.Series(
            encoder.fit_transform(normalized),
            index=y_series.index,
            name=y_series.name,
        )
        return encoded, {i: label for i, label in enumerate(encoder.classes_)}

    if task_mode == "fraud":
        label_map = {label.lower(): label for label in unique_labels}
        positive_label = None
        for hint in FRAUD_POSITIVE_LABEL_HINTS:
            if hint in label_map:
                positive_label = label_map[hint]
                break
        if positive_label is None:
            for label in unique_labels:
                lowered = label.lower()
                if any(hint in lowered for hint in FRAUD_POSITIVE_LABEL_HINTS):
                    positive_label = label
                    break
        if positive_label is not None:
            negative_label = next(label for label in unique_labels if label != positive_label)
            encoded = normalized.map({negative_label: 0, positive_label: 1})
            return pd.Series(encoded, index=y_series.index, name=y_series.name), {
                0: negative_label,
                1: positive_label,
            }

    encoder = LabelEncoder()
    encoded = pd.Series(
        encoder.fit_transform(normalized),
        index=y_series.index,
        name=y_series.name,
    )
    return encoded, {i: label for i, label in enumerate(encoder.classes_)}


def to_numeric_if_possible(series, threshold=0.8):
    if series.dtype != "object":
        return series
    converted = pd.to_numeric(series, errors="coerce")
    if converted.notna().sum() > len(series) * threshold:
        return converted
    return series


def is_likely_free_text(series):
    if series.dtype != "object":
        return False
    sample = series.dropna().astype(str).head(300)
    if sample.empty:
        return False
    avg_length = sample.str.len().mean()
    avg_words = sample.str.split().str.len().mean()
    unique_ratio = sample.nunique() / max(len(sample), 1)
    return avg_length > 30 or avg_words > 5 or unique_ratio > 0.8


def expand_datetime_features(df_input):
    df_expanded = df_input.copy()
    datetime_cols = df_expanded.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns

    for col in datetime_cols:
        dt_series = pd.to_datetime(df_expanded[col], errors="coerce")
        df_expanded[f"{col}_year"] = dt_series.dt.year
        df_expanded[f"{col}_month"] = dt_series.dt.month
        df_expanded[f"{col}_day"] = dt_series.dt.day
        df_expanded[f"{col}_dayofweek"] = dt_series.dt.dayofweek
        df_expanded[f"{col}_hour"] = dt_series.dt.hour
        df_expanded = df_expanded.drop(columns=[col])

    return df_expanded


def compute_text_signal_features(text_series):
    text_series = text_series.fillna("").astype(str)
    lowered = text_series.str.lower()

    features = pd.DataFrame(index=text_series.index)
    features["text_char_count"] = text_series.str.len()
    features["text_word_count"] = text_series.str.split().str.len().fillna(0)
    features["text_digit_count"] = text_series.str.count(r"\d")
    features["text_exclamation_count"] = text_series.str.count(r"!")
    features["text_url_count"] = lowered.str.count(r"http|www\.|bit\.ly")
    features["text_email_count"] = lowered.str.count(r"@")
    features["text_currency_count"] = text_series.str.count(r"[$€£]")
    features["text_uppercase_ratio"] = text_series.apply(
        lambda value: sum(1 for ch in value if ch.isupper()) / max(len(value), 1)
    )

    for feature_name, pattern in FRAUD_SIGNAL_PATTERNS.items():
        features[feature_name] = lowered.str.contains(pattern, regex=True).astype(int)

    return features


def extract_top_class_terms(text_series, labels, top_n=12):
    valid_mask = text_series.fillna("").astype(str).str.strip().ne("")
    if valid_mask.sum() < 3:
        return []

    labels = pd.Series(labels, index=text_series.index)
    if labels.nunique() != 2:
        return []

    vectorizer = TfidfVectorizer(
        max_features=300,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(text_series[valid_mask].astype(str))
    feature_names = np.array(vectorizer.get_feature_names_out())
    label_values = labels[valid_mask].astype(int).to_numpy()

    positive_scores = matrix[label_values == 1].mean(axis=0).A1
    negative_scores = matrix[label_values == 0].mean(axis=0).A1
    lift = positive_scores - negative_scores
    ranked = feature_names[np.argsort(lift)[::-1]]

    useful_terms = []
    for term in ranked:
        normalized = term.replace(" ", "")
        if normalized in ENGLISH_STOP_WORDS:
            continue
        useful_terms.append(term)
        if len(useful_terms) >= top_n:
            break
    return useful_terms


def build_text_reason(row):
    reasons = []
    if row.get("text_signal_urgency_terms", 0):
        reasons.append("urgent language")
    if row.get("text_signal_financial_terms", 0):
        reasons.append("payment or money wording")
    if row.get("text_signal_account_terms", 0):
        reasons.append("account or verification wording")
    if row.get("text_signal_offer_terms", 0):
        reasons.append("promotional or prize wording")
    if row.get("text_signal_action_terms", 0):
        reasons.append("strong call-to-action wording")
    if row.get("text_signal_text_url_count", 0) > 0:
        reasons.append("contains a link")
    if row.get("text_signal_text_uppercase_ratio", 0) > 0.2:
        reasons.append("unusually high uppercase usage")
    if row.get("text_signal_text_exclamation_count", 0) >= 2:
        reasons.append("multiple exclamation marks")

    if not reasons:
        return "Classification driven mostly by learned patterns from the dataset."
    return "Signals detected: " + ", ".join(reasons[:4]) + "."


def prepare_data(df_input, features, target, text_column=None, task_mode=None):
    y = df_input[target].copy()
    structured_features = [col for col in features if col != text_column]
    dropped_text_like_features = []
    cleaned_structured_features = []

    for col in structured_features:
        if is_likely_free_text(df_input[col]):
            dropped_text_like_features.append(col)
        else:
            cleaned_structured_features.append(col)
    structured_features = cleaned_structured_features

    if structured_features:
        X_raw = df_input[structured_features].copy()
    else:
        X_raw = pd.DataFrame(index=df_input.index)

    text_feature_df = None
    top_text_terms = []
    target_mapping = None

    if task_mode == "fraud" and text_column and text_column != "None":
        text_series = df_input[text_column].fillna("").astype(str)
        text_signal_df = compute_text_signal_features(text_series)
        text_signal_df = text_signal_df.add_prefix("text_signal_")

        tfidf = TfidfVectorizer(
            max_features=250,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )
        tfidf_matrix = tfidf.fit_transform(text_series)
        tfidf_feature_names = [
            f"text_term_{make_safe_term_name(term)}"
            for term in tfidf.get_feature_names_out()
        ]
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=df_input.index,
            columns=tfidf_feature_names,
        )

        text_feature_df = pd.concat([text_signal_df, tfidf_df], axis=1)
        st.success(f"NLP analysis enabled for text column: {text_column}")

    if dropped_text_like_features:
        st.info(
            "These free-text-like columns were excluded from structured encoding and should be used through the optional text column selector: "
            + ", ".join(dropped_text_like_features)
        )

    X_raw = expand_datetime_features(X_raw)

    for col in X_raw.columns:
        X_raw[col] = to_numeric_if_possible(X_raw[col])

    if y.dtype == "object":
        y, target_mapping = encode_binary_target(y, task_mode=task_mode)
    else:
        y = pd.to_numeric(y, errors="coerce")

    y = y.dropna()
    if task_mode == "regression":
        y = y.astype(float)

    X_raw = X_raw.loc[y.index]
    if text_feature_df is not None:
        text_feature_df = text_feature_df.loc[y.index]

    if task_mode == "fraud" and text_column and text_column != "None":
        top_text_terms = extract_top_class_terms(
            df_input.loc[y.index, text_column].fillna("").astype(str),
            y,
        )

    num_cols = X_raw.select_dtypes(exclude=["object"]).columns
    cat_cols = X_raw.select_dtypes(include=["object"]).columns

    if len(num_cols) > 0:
        X_raw[num_cols] = X_raw[num_cols].fillna(X_raw[num_cols].median())
    if len(cat_cols) > 0:
        X_raw[cat_cols] = X_raw[cat_cols].fillna("Missing")

    if X_raw.shape[1] > 0:
        X = pd.get_dummies(X_raw, drop_first=True)
    else:
        X = pd.DataFrame(index=y.index)

    if text_feature_df is not None:
        X = pd.concat([X, text_feature_df], axis=1)

    X.columns = clean_feature_names(X.columns)
    X = X.astype(float)

    return (
        X_raw,
        X,
        y,
        text_feature_df,
        top_text_terms,
        target_mapping,
        structured_features,
        dropped_text_like_features,
    )


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        raw_scores = model.decision_function(X_test)
        y_prob = (raw_scores - raw_scores.min()) / max(
            raw_scores.max() - raw_scores.min(), 1e-9
        )

    y_pred_bin = (y_prob >= 0.5).astype(int)
    metrics = {
        "AUC": roc_auc_score(y_test, y_prob),
        "Accuracy": accuracy_score(y_test, y_pred_bin),
        "F1-Score": f1_score(y_test, y_pred_bin, zero_division=0),
        "Recall": recall_score(y_test, y_pred_bin, zero_division=0),
        "Precision": precision_score(y_test, y_pred_bin, zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred_bin)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return metrics, y_prob, y_pred_bin, cm, fpr, tpr


def evaluate_regression(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }
    return metrics, y_pred


def plot_model_comparison(results_df, problem_type):
    fig, ax = plt.subplots(figsize=(9, max(4, len(results_df) * 0.55)))
    if problem_type == "Classification":
        ranked = results_df.sort_values("AUC", ascending=True)
        values = ranked["AUC"]
        colors = ["#1f77b4" if v < values.max() else "#2ca02c" for v in values]
        bars = ax.barh(ranked["Model"], values, color=colors)
        ax.set_xlabel("AUC", fontsize=11)
        ax.set_xlim(0, 1.08)
        for bar, val in zip(bars, values):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
    else:
        ranked = results_df.sort_values("RMSE", ascending=False)
        values = ranked["RMSE"]
        colors = ["#1f77b4" if v > values.min() else "#2ca02c" for v in values]
        bars = ax.barh(ranked["Model"], values, color=colors)
        ax.set_xlabel("RMSE (lower is better)", fontsize=11)
        for bar, val in zip(bars, values):
            ax.text(val + values.max() * 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("Actual Label", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative (0)", "Positive (1)"], fontsize=9)
    ax.set_yticklabels(["Negative (0)", "Positive (1)"], fontsize=9)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color=color, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_roc(fpr, tpr, auc_score, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2,
            label=f"{model_name} (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1f77b4")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def explain_logistic(model, X_columns):
    coef_df = pd.DataFrame(
        {"feature": X_columns, "coefficient": model.coef_[0]}
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coef_df["feature"] = coef_df["feature"].map(prettify_feature_name)
    coef_df["impact_on_positive_class"] = np.where(
        coef_df["coefficient"] >= 0,
        "Pushes toward positive class",
        "Pushes away from positive class",
    )
    st.write("Top Logistic Coefficients")
    st.dataframe(coef_df.head(15))


def explain_linear(model, X_columns):
    coef_df = pd.DataFrame(
        {"feature": X_columns, "coefficient": model.coef_}
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coef_df["feature"] = coef_df["feature"].map(prettify_feature_name)
    coef_df["impact_on_prediction"] = np.where(
        coef_df["coefficient"] >= 0,
        "Increases predicted value",
        "Decreases predicted value",
    )
    st.write("Top Linear Coefficients")
    st.dataframe(coef_df.head(15))


def explain_tree_importance(model, X_columns, title):
    imp_df = pd.DataFrame(
        {"feature": X_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    imp_df["feature"] = imp_df["feature"].map(prettify_feature_name)
    st.write(title)
    st.dataframe(imp_df.head(15))
    st.bar_chart(imp_df.head(15).set_index("feature"))


def explain_permutation_importance(model, X_sample, y_sample, title, problem_type):
    scoring = "r2" if problem_type == "Regression" else "roc_auc"
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=42,
        scoring=scoring,
    )
    imp_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "importance": result.importances_mean,
        }
    ).sort_values("importance", key=np.abs, ascending=False)
    imp_df["feature"] = imp_df["feature"].map(prettify_feature_name)
    st.write(title)
    st.dataframe(imp_df.head(15))
    st.bar_chart(imp_df.head(15).set_index("feature"))


st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader(
    "Upload Data File",
    type=["csv", "xlsx", "xls", "txt", "json", "parquet"],
)

if uploaded_file is None:
    st.info("Upload a dataset to begin.")
    st.stop()

df = load_data(uploaded_file)
if df is None:
    st.stop()

st.subheader("Data Preview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", df.shape[0])
with col2:
    st.metric("Total Columns", df.shape[1])
with col3:
    st.metric("Missing Cells", int(df.isnull().sum().sum()))
st.dataframe(df.head(10))

task_options = [
    "Churn Prediction",
    "Text Classification",
    "Value Prediction",
]
st.sidebar.subheader("Analysis Task")
task = st.sidebar.radio("Choose a task", task_options)

task_mode_map = {
    "Churn Prediction": "churn",
    "Text Classification": "fraud",
    "Value Prediction": "regression",
}
problem_type_map = {
    "Churn Prediction": "Classification",
    "Text Classification": "Classification",
    "Value Prediction": "Regression",
}
task_mode = task_mode_map[task]
problem_type = problem_type_map[task]

st.sidebar.markdown(f"Task type: **{problem_type}**")
if task_mode == "fraud":
    st.sidebar.caption(
        "Text Classification supports both structured fields and free-text fields for content-based analysis."
    )

st.sidebar.subheader("Select Columns")
all_columns = df.columns.tolist()
if task_mode == "regression":
    numeric_targets = [
        col for col in all_columns if pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_targets:
        st.error("Regression requires at least one numeric target column.")
        st.stop()
    target = st.sidebar.selectbox("Target Column", numeric_targets)
else:
    target = st.sidebar.selectbox("Target Column", all_columns)

candidate_feature_columns = [col for col in all_columns if col != target]
text_column = "None"
if task_mode == "fraud":
    text_column = st.sidebar.selectbox(
        "Text Column (optional but recommended)",
        ["None"] + candidate_feature_columns,
    )

default_features = [
    col for col in candidate_feature_columns if col != text_column
][: min(8, len(candidate_feature_columns))]
features = st.sidebar.multiselect(
    "Feature Columns",
    candidate_feature_columns,
    default=default_features,
)

if target in features:
    st.error("Target column should not also be selected as a feature.")
    st.stop()

if task_mode == "fraud":
    if not features and text_column == "None":
        st.warning(
            "Select at least one structured feature or one text column for Text Classification."
        )
        st.stop()
else:
    if not features:
        st.warning("Please select at least one feature column.")
        st.stop()

if task_mode != "regression" and df[target].nunique(dropna=True) != 2:
    st.warning("Classification tasks currently expect a binary target column.")

st.sidebar.subheader("Hyperparameters")
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", 0, 999, 42, 1)

st.sidebar.markdown("---")
st.sidebar.write("**Logistic Regression**")
C_value = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)

st.sidebar.markdown("---")
st.sidebar.write("**Random Forest**")
rf_n_estimators = st.sidebar.slider("RF n_estimators", 50, 300, 100, 10)
rf_max_depth = st.sidebar.slider("RF max_depth", 2, 20, 5, 1)

st.sidebar.markdown("---")
st.sidebar.write("**XGBoost**")
xgb_n_estimators = st.sidebar.slider("XGB n_estimators", 50, 300, 100, 10)
xgb_max_depth = st.sidebar.slider("XGB max_depth", 2, 12, 5, 1)
xgb_learning_rate = st.sidebar.slider("XGB learning_rate", 0.01, 0.30, 0.10, 0.01)

st.sidebar.markdown("---")
st.sidebar.write("**KNN**")
knn_neighbors = st.sidebar.slider("KNN n_neighbors", 1, 50, 5, 1)

st.sidebar.subheader("Model Strategy")
model_mode = st.sidebar.radio(
    "Choose how to run models:",
    ["Auto (Run all models)", "Fast (Recommended set)", "Custom (User select)"],
)

if problem_type == "Classification":
    all_model_list = [
        "Logistic",
        "RandomForest",
        "XGBoost",
        "SVC",
        "KNN",
        "DecisionTree",
        "GradientBoosting",
        "AdaBoost",
        "NaiveBayes",
        "MLP",
    ]
    fast_model_list = ["Logistic", "RandomForest", "XGBoost"]
else:
    all_model_list = [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "RandomForest",
        "XGBoost",
        "DecisionTree",
        "GradientBoosting",
        "ExtraTrees",
        "KNN",
        "MLP",
    ]
    fast_model_list = ["LinearRegression", "RandomForest", "XGBoost"]

if model_mode == "Custom (User select)":
    selected_models = st.sidebar.multiselect(
        "Select models",
        all_model_list,
        default=fast_model_list,
    )
elif model_mode == "Fast (Recommended set)":
    selected_models = fast_model_list
    st.sidebar.info(f"Selected: {', '.join(selected_models)}")
else:
    selected_models = all_model_list
    st.sidebar.info(f"Selected: {', '.join(selected_models)}")

feature_count_for_estimate = len(features) + (1 if text_column != "None" else 0)
est_time = estimate_runtime(
    len(df),
    feature_count_for_estimate,
    selected_models,
    has_text_features=(text_column != "None"),
)
st.sidebar.warning(f"Estimated runtime: {est_time:.2f}s")

st.sidebar.subheader("Execution Decision")
decision = st.sidebar.radio(
    "Status Control:",
    ["Continue", "Stop"],
    help="Select Stop to pause execution while adjusting options.",
)
if decision == "Stop":
    st.sidebar.error("Execution paused. Switch to Continue to enable the Run button.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.info(
    "📸 **Export / Screenshot tip:** Before printing or saving as PDF "
    "(Ctrl+P / Cmd+P), collapse this sidebar using the **arrow button (›)** "
    "at the top-left of the sidebar so charts display at full width."
)

run_button = st.sidebar.button("Run Model")

if not run_button:
    st.info("Click `Run Model` from the sidebar to begin analysis.")
    st.stop()

params = {
    "C_value": C_value,
    "rf_n_estimators": rf_n_estimators,
    "rf_max_depth": rf_max_depth,
    "xgb_n_estimators": xgb_n_estimators,
    "xgb_max_depth": xgb_max_depth,
    "xgb_learning_rate": xgb_learning_rate,
    "knn_neighbors": knn_neighbors,
    "random_state": random_state,
}

(
    X_raw,
    X,
    y,
    text_feature_df,
    top_text_terms,
    target_mapping,
    effective_structured_features,
    dropped_text_like_features,
) = prepare_data(
    df,
    features,
    target,
    text_column=text_column,
    task_mode=task_mode,
)

if len(X) < 2:
    st.error("Not enough valid data to train a model.")
    st.stop()

if task_mode != "regression" and y.nunique() != 2:
    st.error("The selected classification target must contain exactly two classes.")
    st.stop()

split_kwargs = {
    "test_size": test_size,
    "random_state": random_state,
}
if task_mode != "regression":
    split_kwargs["stratify"] = y

X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

progress = st.progress(0)
for i in range(100):
    time.sleep(0.001)
    progress.progress(i + 1)

models = get_models(problem_type, model_mode, selected_models, params)
if not models:
    st.error("No models selected.")
    st.stop()

all_results = []
trained_models = {}
model_artifacts = {}

for name, model in models.items():
    try:
        if task_mode == "regression":
            metrics, y_pred = evaluate_regression(
                model, X_train, X_test, y_train, y_test
            )
            model_artifacts[name] = {"metrics": metrics, "y_pred": y_pred}
        else:
            metrics, y_prob, y_pred, cm, fpr, tpr = evaluate_model(
                model, X_train, X_test, y_train, y_test
            )
            model_artifacts[name] = {
                "metrics": metrics,
                "cm": cm,
                "fpr": fpr,
                "tpr": tpr,
                "y_prob": y_prob,
                "y_pred": y_pred,
            }
        trained_models[name] = model
        row = {"Model": name}
        row.update(metrics)
        all_results.append(row)
    except Exception as exc:
        st.warning(f"Model {name} failed: {exc}")

if not all_results:
    st.error("All selected models failed.")
    st.stop()

results_df = pd.DataFrame(all_results)
if task_mode == "regression":
    results_df = results_df.sort_values("RMSE", ascending=True)
else:
    results_df = results_df.sort_values("AUC", ascending=False)

top3 = results_df.head(3)
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
selected_metrics = model_artifacts[best_model_name]["metrics"]

insight_df = df.loc[X.index].copy()
if task_mode == "regression":
    preds = best_model.predict(X)
    insight_df["predicted_value"] = preds
    insight_df["error"] = pd.to_numeric(insight_df[target], errors="coerce") - preds
else:
    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(X)[:, 1]
    else:
        raw_scores = best_model.decision_function(X)
        probs = (raw_scores - raw_scores.min()) / max(
            raw_scores.max() - raw_scores.min(), 1e-9
        )

    insight_df["prediction_score"] = probs
    insight_df["predicted_label"] = (probs >= 0.5).astype(int)
    insight_df["risk_level"] = pd.cut(
        insight_df["prediction_score"],
        bins=[0, 0.3, 0.7, 1],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    if task_mode == "fraud" and text_feature_df is not None:
        fraud_explain_df = text_feature_df.loc[insight_df.index].copy()
        insight_df["text_risk_summary"] = fraud_explain_df.apply(build_text_reason, axis=1)

tab1, tab2, tab3 = st.tabs(
    ["Model Results", "Record Insights", "Explainability (SHAP)"]
)

with tab1:
    st.subheader("Model Performance")
    if task_mode == "regression":
        st.markdown(f"**Best model selected by RMSE:** `{best_model_name}`")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{selected_metrics['MAE']:.3f}")
        m2.metric("RMSE", f"{selected_metrics['RMSE']:.3f}")
        m3.metric("R2", f"{selected_metrics['R2']:.3f}")
        metric_text = f"RMSE: {selected_metrics['RMSE']:.3f}"
    else:
        st.markdown(f"**Best model selected by AUC:** `{best_model_name}`")
        if target_mapping is not None and 1 in target_mapping:
            st.caption(f"Positive class: `{target_mapping[1]}`")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AUC", f"{selected_metrics['AUC']:.3f}")
        m2.metric("Accuracy", f"{selected_metrics['Accuracy']:.3f}")
        m3.metric("F1", f"{selected_metrics['F1-Score']:.3f}")
        m4.metric("Recall", f"{selected_metrics['Recall']:.3f}")
        m5.metric("Precision", f"{selected_metrics['Precision']:.3f}")
        metric_text = f"AUC: {selected_metrics['AUC']:.3f}"

    st.write("### Recommendation Insight")
    st.markdown(
        f"""
        - Best model: **{best_model_name}**
        - Best performance: **{metric_text}**
        - Top-performing models can be compared below before final deployment.
        """
    )

    if task_mode == "fraud" and text_column != "None" and dropped_text_like_features:
        st.caption(
            "Some long-form text columns were excluded from structured encoding to keep the model general and interpretable."
        )

    st.subheader("All Models Comparison")
    plot_model_comparison(results_df, problem_type)

    if task_mode == "regression":
        st.subheader("All Model Scores")
        score_columns = ["Model", "RMSE", "MAE", "R2"]
    else:
        st.subheader("All Model Scores")
        score_columns = ["Model", "AUC", "Accuracy", "F1-Score", "Recall", "Precision"]

    display_results_df = results_df[score_columns].copy()
    numeric_cols = display_results_df.select_dtypes(include=[np.number]).columns
    display_results_df[numeric_cols] = display_results_df[numeric_cols].round(3)
    top3_display_df = display_results_df[display_results_df["Model"].isin(top3["Model"])]

    if task_mode == "regression":
        styled_df = display_results_df.style
        styled_df = styled_df.highlight_min(
            subset=pd.IndexSlice[top3_display_df.index, ["RMSE", "MAE"]],
            axis=0,
        )
        styled_df = styled_df.highlight_max(
            subset=pd.IndexSlice[top3_display_df.index, ["R2"]],
            axis=0,
        )
    else:
        top3_metric_cols = ["AUC", "Accuracy", "F1-Score", "Recall", "Precision"]
        styled_df = display_results_df.style.highlight_max(
            subset=pd.IndexSlice[top3_display_df.index, top3_metric_cols],
            axis=0,
        )

    st.dataframe(styled_df.format(precision=3), use_container_width=True)

    if problem_type == "Classification":
        st.write("### Best Model Diagnostics")
        plot_confusion_matrix(model_artifacts[best_model_name]["cm"], best_model_name)
        plot_roc(
            model_artifacts[best_model_name]["fpr"],
            model_artifacts[best_model_name]["tpr"],
            model_artifacts[best_model_name]["metrics"]["AUC"],
            best_model_name,
        )

with tab2:
    st.subheader("Record Insights")
    if task_mode == "churn":
        st.write("Top high-risk records:")
        st.dataframe(
            insight_df.sort_values("prediction_score", ascending=False).head(10)
        )
    elif task_mode == "fraud":
        st.write("Top flagged records:")
        if text_column in insight_df.columns:
            insight_df["text_preview"] = (
                insight_df[text_column]
                .fillna("")
                .astype(str)
                .str.slice(0, 140)
            )
        display_cols = []
        if "text_preview" in insight_df.columns:
            display_cols.append("text_preview")
        display_cols.extend(
            [
                col
                for col in [
                    "prediction_score",
                    "risk_level",
                    "text_risk_summary",
                ]
                if col in insight_df.columns
            ]
        )
        remaining_feature_cols = [
            col for col in effective_structured_features if col in insight_df.columns
        ][:3]
        display_cols.extend(
            [col for col in remaining_feature_cols if col not in display_cols]
        )
        if not display_cols:
            display_cols = ["prediction_score", "risk_level"]
        st.dataframe(
            insight_df.sort_values("prediction_score", ascending=False)[display_cols].head(15)
        )
    else:
        st.write("Predicted values:")
        st.dataframe(
            insight_df[
                [target]
                + [col for col in effective_structured_features if col in insight_df.columns]
                + ["predicted_value"]
            ].head(10)
        )
        st.markdown("### Prediction Error Analysis")
        st.dataframe(
            insight_df[[target, "predicted_value", "error"]]
            .sort_values("error", key=np.abs, ascending=False)
            .head(10)
        )

    st.markdown("---")
    st.download_button(
        "Download Full Prediction Results",
        insight_df.to_csv(index=False),
        "predictions.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Model Explainability")
    st.markdown(f"**Explaining Best Model:** `{best_model_name}`")

    if best_model_name in [
        "RandomForest",
        "XGBoost",
        "GradientBoosting",
        "ExtraTrees",
        "DecisionTree",
    ]:
        with st.spinner("Calculating SHAP values..."):
            try:
                sample_X = X_test.sample(min(200, len(X_test)), random_state=42)
                sample_X = sample_X.apply(pd.to_numeric, errors="coerce").fillna(0)
                if task_mode == "regression":
                    explainer = shap.Explainer(best_model.predict, sample_X)
                else:
                    explainer = shap.Explainer(best_model)
                shap_values = explainer(sample_X)
                plt.figure(figsize=(14, 5))
                shap.summary_plot(
                    shap_values,
                    sample_X,
                    max_display=15,
                    plot_type="dot",
                    show=False,
                )
                plt.title(f"SHAP Feature Impact ({best_model_name})", fontsize=14)
                plt.xlabel("Impact on Model Output (SHAP value)", fontsize=11)
                plt.ylabel("Features", fontsize=11)
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as exc:
                st.warning(f"SHAP failed: {exc}")

    if best_model_name == "Logistic":
        if target_mapping is not None and 1 in target_mapping:
            st.caption(
                f"Positive coefficients push predictions toward `{target_mapping[1]}`."
            )
        explain_logistic(best_model, X.columns)
    elif best_model_name in ["LinearRegression", "Ridge", "Lasso"]:
        explain_linear(best_model, X.columns)
    elif best_model_name in [
        "RandomForest",
        "XGBoost",
        "GradientBoosting",
        "ExtraTrees",
        "DecisionTree",
        "AdaBoost",
    ]:
        explain_tree_importance(best_model, X.columns, f"{best_model_name} Importance")
    elif best_model_name in ["SVC", "KNN", "MLP", "SVR", "NaiveBayes"]:
        sample_X = X_test.sample(min(300, len(X_test)), random_state=42)
        sample_y = y_test.loc[sample_X.index]
        explain_permutation_importance(
            best_model,
            sample_X,
            sample_y,
            f"{best_model_name} Permutation Importance",
            problem_type,
        )
    else:
        st.info("Visual importance for this model type is not pre-configured.")

    st.write("### Interpretation Notes")
    st.markdown(
        """
        - **SHAP** shows how each feature pushes predictions up or down.
        - **Importance** shows which variables contribute most overall.
        - **Permutation importance** measures how much model quality drops when a feature is shuffled.
        - **Text signal summary** gives a human-readable explanation based on detected linguistic patterns in the text column.
        """
    )
