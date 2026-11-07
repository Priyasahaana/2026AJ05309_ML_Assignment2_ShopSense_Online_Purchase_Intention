
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "Revenue"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="ShopSense | Purchase Intelligence", page_icon="🛍️", layout="wide")
st.markdown(
    "<style>.stApp{background:linear-gradient(180deg,#f7f9fc 0%,#fff 45%);}"
    ".main-title{font-size:2.45rem;font-weight:800;color:#172554;margin-bottom:0;}"
    ".subtitle{color:#475569;font-size:1.02rem;margin-top:.2rem;}"
    "div[data-testid='stMetric']{background:#fff;border:1px solid #e2e8f0;"
    "padding:14px 16px;border-radius:12px;}</style>",
    unsafe_allow_html=True
)
st.markdown("<div class='main-title'>ShopSense</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Online Purchase Intention Analytics · "
    "Compare five classifiers on held-out sessions</div>",
    unsafe_allow_html=True
)
st.write("")
st.info(
    "Upload a labelled test CSV containing the session features and Revenue target. "
    "The selected saved model evaluates the file; no model is retrained."
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

with st.sidebar:
    st.header("Evaluation controls")
    selected_model = st.selectbox("Choose classifier", list(MODEL_FILES.keys()))
    uploaded_file = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])
    st.caption("Expected target: Revenue (True/False or 1/0).")
    st.divider()
    st.caption("Saved pipelines are loaded from the repository's model/ folder.")

if uploaded_file is None:
    st.subheader("How to use this dashboard")
    st.markdown(
        "1. Select a classifier in the sidebar.\n"
        "2. Upload `test_data.csv` from the project folder.\n"
        "3. Review the six metrics, confusion matrix, and class-wise report."
    )
    st.caption("The repository includes a sample held-out test file for evaluation.")
    st.stop()

try:
    test_df = pd.read_csv(uploaded_file)
    if TARGET not in test_df.columns:
        raise ValueError("The uploaded CSV must include the target column Revenue.")

    y_raw = test_df[TARGET]
    if pd.api.types.is_bool_dtype(y_raw):
        y_true = y_raw.astype(int)
    else:
        target_map = {True: 1, False: 0, "true": 1, "false": 0, "1": 1, "0": 0, 1: 1, 0: 0}
        y_true = y_raw.map(
            lambda v: target_map.get(v, target_map.get(str(v).strip().lower(), np.nan))
        )
        if y_true.isna().any():
            raise ValueError("Revenue values must be True/False or 1/0.")
        y_true = y_true.astype(int)

    X_uploaded = test_df.drop(columns=[TARGET])
    model_path = MODEL_DIR / MODEL_FILES[selected_model]
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path.name}")

    pipeline = load_model(str(model_path))
    expected_features = list(pipeline.feature_names_in_)
    missing = sorted(set(expected_features) - set(X_uploaded.columns))
    extra = sorted(set(X_uploaded.columns) - set(expected_features))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    if extra:
        st.warning(f"Extra columns will be ignored: {extra}")
    X_uploaded = X_uploaded[expected_features]

    y_pred = pipeline.predict(X_uploaded)
    y_score = pipeline.predict_proba(X_uploaded)[:, 1]
    auc_value = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
    if np.isnan(auc_value):
        st.warning("ROC-AUC is undefined because the uploaded target contains only one class.")

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC-AUC": auc_value,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    st.subheader(f"Results · {selected_model}")
    st.caption(f"Evaluated {len(test_df):,} labelled sessions.")
    metric_cols = st.columns(3)
    for i, (name, value) in enumerate(metrics.items()):
        with metric_cols[i % 3]:
            st.metric(name, "N/A" if pd.isna(value) else f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("### Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        ConfusionMatrixDisplay(cm, display_labels=["No purchase", "Purchase"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(selected_model)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("### Classification report")
        report = classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["No purchase", "Purchase"],
            output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    with st.expander("Preview uploaded test data"):
        st.dataframe(test_df.head(20), use_container_width=True)

except Exception as exc:
    st.error(f"Evaluation could not be completed: {exc}")
    st.caption("Check that the CSV has the expected 17 input columns and a valid Revenue target.")
