"""
app.py - Dry Bean Variety Classifier (ML Assignment 2)
------------------------------------------------------
Interactive Streamlit front-end for the five pre-trained classifiers.

Features required by the assignment:
  a. CSV upload of test data             -> sidebar uploader
  b. Model selection dropdown            -> sidebar selectbox
  c. Display of evaluation metrics       -> "Selected model" tab
  d. Confusion matrix + classif. report  -> "Selected model" tab

The app ONLY loads the saved pipelines from model/ and calls predict /
predict_proba. Nothing is ever re-trained on the uploaded data.

Run locally:  streamlit run app.py
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from ml_utils import (
    METADATA_PATH,
    METRIC_COLUMNS,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    TARGET_COL,
    TEST_DATA_PATH,
    compute_metrics,
)

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")

# Small amount of custom styling so the UI is not the stock template.
st.markdown(
    """
    <style>
    .hero {padding: 1.1rem 1.4rem; border-radius: 12px;
           background: linear-gradient(90deg, #4b2e14 0%, #8a5a2b 60%, #c9a227 100%);
           color: #fff8e7; margin-bottom: 1rem;}
    .hero h1 {color: #fff8e7; margin: 0; font-size: 2rem;}
    .hero p  {margin: .3rem 0 0 0; opacity: .9;}
    div[data-testid="stMetric"] {background: #faf5ea; border: 1px solid #e6d8b8;
                                 border-radius: 10px; padding: .6rem .8rem;}
    </style>
    <div class="hero">
      <h1>🫘 Dry Bean Variety Classifier Lab</h1>
      <p>Evaluate five pre-trained models on 7 dry-bean varieties using 16
         shape features extracted from grain images (UCI Dry Bean dataset).</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached loaders - models are read from disk once per server process.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading saved models ...")
def load_models():
    missing = [f for f in MODEL_FILES.values() if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files in model/: {missing}. Run train_models.py first.")
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_metadata():
    return json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}


@st.cache_data
def load_reference_metrics():
    return pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()


def get_feature_names(models):
    """Feature names exactly as seen during training (same for all pipelines)."""
    first = next(iter(models.values()))
    return list(first.feature_names_in_)


def validate_and_prepare(df, feature_names):
    """Check the uploaded frame and return clean (X, y) or raise ValueError.

    * Column names are stripped of whitespace.
    * Extra columns are ignored; features are re-ordered to the training order.
    * Non-numeric feature values / missing values are dropped with a warning.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"The uploaded CSV has no '{TARGET_COL}' (target) column. The assignment "
            "requires the test file to contain the true labels so metrics can be computed."
        )
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"The uploaded CSV is missing required feature column(s): {missing}")

    X = df[feature_names].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET_COL].astype(str).str.strip()

    valid = X.notna().all(axis=1) & df[TARGET_COL].notna()
    dropped = int((~valid).sum())
    if dropped:
        st.warning(f"{dropped} row(s) with missing / non-numeric values were skipped.")
    X, y = X[valid], y[valid]

    if X.empty:
        raise ValueError("No valid rows left after cleaning the uploaded file.")
    return X, y


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return compute_metrics(y, y_pred, y_proba, model.classes_), y_pred, y_proba


def fmt(value):
    return "N/A" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.4f}"


# --------------------------------------------------------------------------- #
# Load artefacts
# --------------------------------------------------------------------------- #
try:
    models = load_models()
except Exception as exc:  # show a friendly message instead of a stack trace
    st.error(f"Could not load the trained models: {exc}")
    st.stop()

feature_names = get_feature_names(models)
classes = list(next(iter(models.values())).classes_)
metadata = load_metadata()

# --------------------------------------------------------------------------- #
# Sidebar - data upload + model selection
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1️⃣ Test data")
    uploaded = st.file_uploader(
        "Upload test CSV (features + Class column)",
        type=["csv"],
        help="Upload only test data - the free Streamlit tier has limited memory.",
    )
    if TEST_DATA_PATH.exists():
        st.download_button(
            "⬇️ Download sample test_data.csv",
            data=TEST_DATA_PATH.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
    use_bundled = False
    if uploaded is None and TEST_DATA_PATH.exists():
        use_bundled = st.checkbox("No file? Use the bundled test_data.csv", value=False)

    st.header("2️⃣ Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))
    normalise_cm = st.toggle("Normalise confusion matrix (row %)", value=False)

    st.divider()
    st.caption(
        f"Models expect {len(feature_names)} features and predict one of "
        f"{len(classes)} classes: {', '.join(classes)}."
    )

# --------------------------------------------------------------------------- #
# Read + validate the data
# --------------------------------------------------------------------------- #
if uploaded is not None:
    raw_df, source = pd.read_csv(uploaded), f"uploaded file **{uploaded.name}**"
elif use_bundled:
    raw_df, source = pd.read_csv(TEST_DATA_PATH), "bundled **test_data.csv**"
else:
    raw_df, source = None, None

tab_model, tab_compare, tab_about = st.tabs(
    ["📊 Selected model", "🏁 Compare all models", "ℹ️ Dataset & training"]
)

if raw_df is None:
    with tab_model:
        st.info(
            "👈 Upload a test CSV in the sidebar (or tick *Use the bundled test_data.csv*) "
            "to evaluate the models. The file must contain these columns:"
        )
        st.code(", ".join(feature_names + [TARGET_COL]))
    with tab_compare:
        st.info("Upload test data to compare all five models on it.")
else:
    try:
        X, y = validate_and_prepare(raw_df, feature_names)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    unknown = sorted(set(y) - set(classes))
    if unknown:
        # Labels the models have never seen can never be predicted correctly;
        # drop them so they don't silently distort the metrics.
        st.warning(f"Rows with unknown class label(s) {unknown} were excluded.")
        keep = y.isin(classes)
        X, y = X[keep], y[keep]
        if X.empty:
            st.error("No rows with a known class label remain.")
            st.stop()

    # ------------------------------------------------------------------ #
    # Tab 1 - the selected model
    # ------------------------------------------------------------------ #
    with tab_model:
        st.markdown(f"Evaluating **{model_name}** on {source} — **{len(X)}** rows.")
        model = models[model_name]
        metrics, y_pred, y_proba = evaluate(model, X, y)

        cols = st.columns(len(METRIC_COLUMNS))
        for col, metric in zip(cols, METRIC_COLUMNS):
            col.metric(metric, fmt(metrics[metric]))
        st.caption(
            "Precision / Recall / F1 / AUC use a class-support **weighted** average "
            "(AUC is One-vs-Rest). MCC is the multi-class Matthews correlation."
        )

        left, right = st.columns([1.15, 1])
        with left:
            st.subheader("Confusion matrix")
            cm = confusion_matrix(y, y_pred, labels=classes)
            if normalise_cm:
                # Row-normalise; rows with no true samples stay at 0.
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_plot = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
                fmt_code = ".2f"
            else:
                cm_plot, fmt_code = cm, "d"
            fig, ax = plt.subplots(figsize=(7, 5.5))
            sns.heatmap(cm_plot, annot=True, fmt=fmt_code, cmap="YlOrBr", cbar=False,
                        xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title(f"{model_name}")
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.subheader("Classification report")
            report = classification_report(y, y_pred, labels=classes, output_dict=True, zero_division=0)
            # "accuracy" is a single number, not a per-class dict; keeping it
            # would create a malformed row, so it is shown as a caption instead.
            overall_acc = report.pop("accuracy", None)
            report_df = pd.DataFrame(report).T
            report_df["support"] = report_df["support"].astype(int)
            st.dataframe(report_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                                 "f1-score": "{:.4f}"}))
            if overall_acc is not None:
                st.caption(f"Overall accuracy: {overall_acc:.4f}")

        with st.expander("🔎 Predictions preview & download"):
            preds_df = X.copy()
            preds_df["True Class"] = y.values
            preds_df["Predicted Class"] = y_pred
            preds_df["Confidence"] = y_proba.max(axis=1).round(4)
            preds_df["Correct"] = preds_df["True Class"] == preds_df["Predicted Class"]
            st.dataframe(preds_df.head(200))
            st.download_button(
                "⬇️ Download all predictions (CSV)",
                preds_df.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{MODEL_FILES[model_name].replace('.joblib', '')}.csv",
                mime="text/csv",
            )

    # ------------------------------------------------------------------ #
    # Tab 2 - all five models side by side on the same data
    # ------------------------------------------------------------------ #
    with tab_compare:
        st.markdown(f"All five models evaluated on {source} — **{len(X)}** rows.")
        rows = []
        for name, mdl in models.items():
            m, _, _ = evaluate(mdl, X, y)
            rows.append({"ML Model Name": name, **m})
        comparison = pd.DataFrame(rows).set_index("ML Model Name")

        st.dataframe(
            comparison.style.format("{:.4f}", na_rep="N/A").highlight_max(axis=0, color="#f3d98b")
        )
        best = comparison["F1"].idxmax()
        st.success(f"🏆 Best model on this data by F1: **{best}** "
                   f"(F1 = {comparison.loc[best, 'F1']:.4f}, MCC = {comparison.loc[best, 'MCC']:.4f})")

        metric_to_plot = st.radio("Metric to plot", METRIC_COLUMNS, horizontal=True, index=4)
        fig, ax = plt.subplots(figsize=(9, 3.8))
        plot_data = comparison[metric_to_plot].sort_values()
        bars = ax.barh(plot_data.index, plot_data.values, color="#a0692f")
        ax.bar_label(bars, fmt="%.4f", padding=3)
        # Zoom the x-axis so small differences between strong models are visible.
        low = np.nanmin(plot_data.values)
        ax.set_xlim(max(0.0, low - 0.1) if not np.isnan(low) else 0, 1.02)
        ax.set_xlabel(metric_to_plot)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------------------- #
# Tab 3 - static information about the dataset and training run
# ------------------------------------------------------------------------- #
with tab_about:
    st.subheader("Dataset")
    st.markdown(
        "**UCI Dry Bean Dataset** (Koklu & Ozkan, 2020). Images of 13,611 grains of 7 "
        "registered dry-bean varieties were segmented and 16 features were extracted: 12 "
        "dimension features (area, perimeter, axis lengths, ...) and 4 shape factors. "
        "Task: multi-class classification of the bean variety (`Class`)."
    )
    if metadata:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows (after cleaning)", metadata.get("n_rows_after_cleaning", "-"))
        c2.metric("Features", len(metadata.get("features", [])))
        c3.metric("Train rows", metadata.get("n_train", "-"))
        c4.metric("Test rows", metadata.get("n_test", "-"))

        dist = pd.Series(metadata.get("class_distribution", {}), name="samples")
        if not dist.empty:
            st.markdown("**Class distribution (full cleaned dataset)**")
            st.bar_chart(dist)

        st.subheader("Tuned hyper-parameters (5-fold CV on the training set)")
        hp = pd.DataFrame(
            [{"Model": k, "Best parameters": json.dumps(v["best_params"]),
              "CV F1 (weighted)": v["cv_f1_weighted"]} for k, v in metadata.get("models", {}).items()]
        )
        st.dataframe(hp, hide_index=True)
        st.caption(f"Trained with: {metadata.get('versions', {})}")

    ref = load_reference_metrics()
    if not ref.empty:
        st.subheader("Reference hold-out metrics (computed at training time)")
        st.dataframe(ref.set_index("ML Model Name").style.format("{:.4f}"))
