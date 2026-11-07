
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_curve
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

COLORS = {
    "Logistic Regression": "#2563EB",
    "Decision Tree": "#0D9488",
    "K-Nearest Neighbors": "#D97706",
    "Gaussian Naive Bayes": "#9333EA",
    "Support Vector Machine": "#E05252",
}

st.set_page_config(
    page_title="ShopSense | Purchase Analytics",
    page_icon="🛍️",
    layout="wide"
)

# -------------------- DASHBOARD DESIGN --------------------
st.markdown("""
<style>
.stApp {
    background: #F3F6FB;
    color: #172B4D;
}
[data-testid="stAppViewContainer"] {
    background: #F3F6FB;
}
[data-testid="stHeader"] {
    background: rgba(243,246,251,0.95);
}
[data-testid="stSidebar"] {
    background: #14243B;
}
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] input {
    color: #F8FAFC !important;
}
/* Keep dropdown background dark in both light and dark themes */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #26364B !important;
    border: 1px solid #52647A !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #F8FAFC !important;
    color: #F8FAFC !important;
}
/* Keep dropdown options dark when the menu is opened */
[data-baseweb="popover"] [role="listbox"] {
    background-color: #26364B !important;
}
[data-baseweb="popover"] [role="option"] {
    background-color: #26364B !important;
    color: #F8FAFC !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background-color: #36516F !important;
    color: #FFFFFF !important;
}
/* Keep file uploader background dark in both themes */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background-color: #26364B !important;
    border: 1px solid #52647A !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section * {
    color: #F8FAFC !important;
}
/* Keep upload button dark with readable text */
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background-color: #26364B !important;
    color: #FFFFFF !important;
    border: 1px solid #52647A !important;
}
/* Match classifier dropdown background to file uploader */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #26364B !important;
    background-color: #26364B !important;
    border: 1px solid #52647A !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div > div {
    background: #26364B !important;
    background-color: #26364B !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] input {
    background: transparent !important;
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #F8FAFC !important;
    color: #F8FAFC !important;
}
.main-title {
    color: #14243B;
    font-size: 2.65rem;
    font-weight: 850;
    letter-spacing: -1.2px;
    margin-bottom: 0;
}
.subtitle {
    color: #52647A;
    font-size: 1.08rem;
    margin-top: 5px;
    margin-bottom: 20px;
}
.section-title {
    color: #14243B;
    font-size: 1.35rem;
    font-weight: 750;
    margin-top: 12px;
}
.info-panel {
    background: #E5F0FF;
    border-left: 5px solid #2563EB;
    padding: 15px 18px;
    border-radius: 9px;
    color: #193B65;
    margin: 14px 0 22px 0;
}
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE5F0;
    border-radius: 13px;
    padding: 16px 18px;
    box-shadow: 0 3px 12px rgba(30,55,90,0.05);
}
div[data-testid="stMetricLabel"] {
    color: #52647A;
    font-weight: 650;
}
div[data-testid="stMetricValue"] {
    color: #14243B;
    font-weight: 800;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}
.stTabs [data-baseweb="tab"] {
    color: #263B58;
    font-weight: 650;
}
div[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border-radius: 10px;
}
h1, h2, h3, p, label {
    color: #172B4D;
}
.stCaption {
    color: #64748B !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown(
    '<div class="main-title">🛍️ ShopSense</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Online Purchase Intention Analytics | '
    'An interactive comparison of five machine learning classifiers</div>',
    unsafe_allow_html=True
)
st.markdown("""
<div class="info-panel">
<b>Dashboard purpose:</b> Evaluate saved machine learning pipelines on
labelled website-session data. Compare their predictive performance,
identify the strongest classifier, and understand purchase behaviour.
<br><br>
<b>Important:</b> Models are loaded from the repository. This dashboard
does not retrain any model.
</div>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

with st.sidebar:
    st.title("Evaluation controls")
    st.markdown("---")
    selected_model = st.selectbox(
        "Choose a classifier",
        list(MODEL_FILES.keys())
    )
    uploaded_file = st.file_uploader(
        "Upload labelled test data (CSV)",
        type=["csv"]
    )
    st.markdown("---")
    st.markdown("### Dataset requirements")
    st.caption("• 17 input features")
    st.caption("• Target column: Revenue")
    st.caption("• Revenue values: 0/1 or False/True")
    st.markdown("---")
    st.caption("Five saved pipelines are available in the model/ folder.")

if uploaded_file is None:
    st.markdown("## Welcome to the ShopSense dashboard")
    st.write(
        "Upload the labelled `test_data.csv` file using the sidebar "
        "to begin evaluating the five saved classifiers."
    )
    st.info(
        "The uploaded file must contain all 17 input features and "
        "the Revenue target column."
    )
    st.markdown("### What you can explore")
    col1, col2, col3 = st.columns(3)
    col1.metric("Classifiers", "5")
    col2.metric("Evaluation metrics", "6")
    col3.metric("Input features", "17")
    st.stop()

# -------------------- DATA VALIDATION --------------------
try:
    test_df = pd.read_csv(uploaded_file)

    if TARGET not in test_df.columns:
        raise ValueError("The CSV must contain a Revenue target column.")

    y_raw = test_df[TARGET]

    if pd.api.types.is_bool_dtype(y_raw):
        y_true = y_raw.astype(int)
    else:
        target_map = {
            True: 1, False: 0,
            "true": 1, "false": 0,
            "1": 1, "0": 0, 1: 1, 0: 0
        }
        y_true = y_raw.map(
            lambda value: target_map.get(
                value,
                target_map.get(str(value).strip().lower(), np.nan)
            )
        )
        if y_true.isna().any():
            raise ValueError("Revenue must contain only 0/1 or True/False.")
        y_true = y_true.astype(int)

    if len(np.unique(y_true)) != 2:
        raise ValueError("The test data must contain both Revenue classes.")

    X_uploaded = test_df.drop(columns=[TARGET])

    # -------------------- EVALUATE ALL FIVE MODELS --------------------
    comparison_rows = []
    predictions = {}
    scores = {}
    pipelines = {}

    for model_name, model_file in MODEL_FILES.items():
        model_path = MODEL_DIR / model_file

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_file}")

        pipeline = load_model(str(model_path))
        expected_features = list(pipeline.feature_names_in_)

        missing = sorted(set(expected_features) - set(X_uploaded.columns))
        if missing:
            raise ValueError(f"Missing input features: {missing}")

        model_input = X_uploaded[expected_features]
        model_predictions = pipeline.predict(model_input)
        model_scores = pipeline.predict_proba(model_input)[:, 1]

        predictions[model_name] = model_predictions
        scores[model_name] = model_scores
        pipelines[model_name] = pipeline

        comparison_rows.append({
            "ML Model Name": model_name,
            "Accuracy": accuracy_score(y_true, model_predictions),
            "AUC": roc_auc_score(y_true, model_scores),
            "Precision": precision_score(
                y_true, model_predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_true, model_predictions, zero_division=0
            ),
            "F1": f1_score(
                y_true, model_predictions, zero_division=0
            ),
            "MCC": matthews_corrcoef(y_true, model_predictions)
        })

    comparison_df = pd.DataFrame(comparison_rows)
    selected_metrics = comparison_df[
        comparison_df["ML Model Name"] == selected_model
    ].iloc[0]
    y_pred = predictions[selected_model]

    # -------------------- KEY PERFORMANCE CARDS --------------------
    st.markdown("## Overall evaluation summary")
    st.caption(
        f"Results based on {len(test_df):,} labelled website sessions."
    )

    metric_columns = st.columns(3)
    metric_names = [
        ("Accuracy", "Accuracy"),
        ("AUC", "ROC-AUC"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1", "F1-score"),
        ("MCC", "Matthews correlation")
    ]

    for index, (column_name, display_name) in enumerate(metric_names):
        with metric_columns[index % 3]:
            st.metric(
                display_name,
                f"{selected_metrics[column_name]:.4f}"
            )

    st.markdown(
        f"### Selected classifier: {selected_model}"
    )

    # -------------------- MODEL COMPARISON TABLE --------------------
    st.markdown("## Comparison of all five classifiers")
    st.caption(
        "Every classifier is evaluated on the same uploaded test dataset."
    )

    display_df = comparison_df.copy()
    numeric_columns = [
        "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"
    ]
    styled_df = display_df.style.format(
        {column: "{:.4f}" for column in numeric_columns}
    ).background_gradient(
        subset=numeric_columns, cmap="Blues", vmin=0, vmax=1
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    # -------------------- AUTOMATIC INTERPRETATION --------------------
    # SHOPSENSE_READABLE_INSIGHT_CARDS
    best_f1 = comparison_df.loc[comparison_df["F1"].idxmax()]
    best_recall = comparison_df.loc[comparison_df["Recall"].idxmax()]
    best_precision = comparison_df.loc[comparison_df["Precision"].idxmax()]
    best_auc = comparison_df.loc[comparison_df["AUC"].idxmax()]
    st.markdown("## What do these results tell us?")
    insight1, insight2 = st.columns(2)
    with insight1:
        st.markdown(
            f"""
            <div style="background-color:#FFFFFF;border:1px solid #C6D9F5;border-left:5px solid #2563EB;border-radius:12px;padding:20px;margin-bottom:14px;min-height:82px;">
                <span style="color:#14243B;font-weight:800;">Best F1-score:</span>
                <span style="color:#334155;"> {best_f1['ML Model Name']} ({best_f1['F1']:.4f}). F1 balances precision and recall.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div style="background-color:#FFFFFF;border:1px solid #C6D9F5;border-left:5px solid #2563EB;border-radius:12px;padding:20px;margin-bottom:14px;min-height:82px;">
                <span style="color:#14243B;font-weight:800;">Best ROC-AUC:</span>
                <span style="color:#334155;"> {best_auc['ML Model Name']} ({best_auc['AUC']:.4f}). This model has the strongest ranking ability.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with insight2:
        st.markdown(
            f"""
            <div style="background-color:#FFFFFF;border:1px solid #C6D9F5;border-left:5px solid #2563EB;border-radius:12px;padding:20px;margin-bottom:14px;min-height:82px;">
                <span style="color:#14243B;font-weight:800;">Best recall:</span>
                <span style="color:#334155;"> {best_recall['ML Model Name']} ({best_recall['Recall']:.4f}). It identifies the largest share of actual purchases.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div style="background-color:#FFFFFF;border:1px solid #C6D9F5;border-left:5px solid #2563EB;border-radius:12px;padding:20px;margin-bottom:14px;min-height:82px;">
                <span style="color:#14243B;font-weight:800;">Best precision:</span>
                <span style="color:#334155;"> {best_precision['ML Model Name']} ({best_precision['Precision']:.4f}). Its positive purchase predictions are the most precise.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        

    # -------------------- VISUAL ANALYTICS --------------------
    st.markdown("## Visual model analysis")

    chart_tab, roc_tab, detail_tab = st.tabs([
        "📊 Metric comparison",
        "📈 ROC curves",
        "🔍 Selected model details"
    ])

    with chart_tab:
        st.markdown("### Performance comparison across six metrics")
        st.caption(
            "Higher values indicate stronger performance. "
            "Compare each metric separately."
        )

        metric_labels = [
            "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"
        ]
        x = np.arange(len(metric_labels))
        width = 0.15

        fig, ax = plt.subplots(figsize=(13, 5.5))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        for index, model_name in enumerate(MODEL_FILES.keys()):
            row = comparison_df[
                comparison_df["ML Model Name"] == model_name
            ].iloc[0]
            values = [row[metric] for metric in metric_labels]
            offset = (index - 2) * width

            ax.bar(
                x + offset,
                values,
                width,
                label=model_name,
                color=COLORS[model_name],
                edgecolor="white",
                linewidth=0.5
            )

        ax.set_ylabel("Metric score", fontsize=11, color="#334155")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.set_title(
            "Five-model performance comparison",
            fontsize=15,
            fontweight="bold",
            color="#14243B",
            pad=16
        )
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=3,
            frameon=False
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Note: MCC is a correlation-based measure. "
            "All six scores are displayed on a common -1 to 1 "
            "conceptual scale, but this chart starts at zero for "
            "readability. The plotted MCC values are the actual values."
        )

    with roc_tab:
        st.markdown("### Receiver Operating Characteristic (ROC)")
        st.caption(
            "The ROC curve shows the trade-off between true positive "
            "rate and false positive rate at different thresholds. "
            "A curve closer to the top-left generally indicates better "
            "classification performance."
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        for model_name in MODEL_FILES.keys():
            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_true, scores[model_name]
            )
            auc_score = comparison_df.loc[
                comparison_df["ML Model Name"] == model_name, "AUC"
            ].iloc[0]

            ax.plot(
                false_positive_rate,
                true_positive_rate,
                linewidth=2.4,
                color=COLORS[model_name],
                label=f"{model_name} (AUC = {auc_score:.3f})"
            )

        ax.plot(
            [0, 1], [0, 1],
            linestyle="--",
            color="#94A3B8",
            linewidth=1.5,
            label="Random classifier"
        )
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(
            "ROC curve comparison",
            fontsize=15,
            fontweight="bold",
            color="#14243B",
            pad=14
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", frameon=True, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with detail_tab:
        st.markdown(f"### Confusion matrix: {selected_model}")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        left, right = st.columns([1.1, 1])

        with left:
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor("#FFFFFF")
            ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title(
                "Actual vs predicted purchases",
                fontsize=13,
                fontweight="bold",
                color="#14243B"
            )
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["No purchase", "Purchase"])
            ax.set_yticklabels(["No purchase", "Purchase"])
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class")

            threshold = cm.max() / 2
            for row in range(2):
                for column in range(2):
                    ax.text(
                        column, row, f"{cm[row, column]:,}",
                        ha="center", va="center",
                        fontsize=15, fontweight="bold",
                        color="white" if cm[row, column] > threshold
                        else "#14243B"
                    )

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.markdown("#### How to read the confusion matrix")
            st.write(
                "**True Negative (top-left):** Correctly predicted "
                "no-purchase sessions."
            )
            st.write(
                "**False Positive (top-right):** Predicted purchase, "
                "but no purchase occurred."
            )
            st.write(
                "**False Negative (bottom-left):** Actual purchase "
                "sessions missed by the model."
            )
            st.write(
                "**True Positive (bottom-right):** Correctly identified "
                "purchase sessions."
            )

        st.markdown("### Classification report")
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["No purchase", "Purchase"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report).T.round(4)
        st.dataframe(report_df, use_container_width=True)

    # -------------------- DATA PREVIEW --------------------
    with st.expander("View uploaded test data"):
        st.dataframe(test_df.head(20), use_container_width=True)
        st.caption(
            f"Showing the first 20 rows of {len(test_df):,} uploaded rows."
        )

    st.markdown("---")
    st.caption(
        "ShopSense | Online Purchase Intention Analytics | "
        "Five saved classifiers | Evaluation without retraining"
    )

except Exception as exc:
    st.error(f"Evaluation could not be completed: {exc}")
    st.caption(
        "Check that the uploaded CSV contains all 17 expected input "
        "features and a valid Revenue target."
    )
