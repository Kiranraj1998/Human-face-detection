# app.py
import io
import json
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ------------------------------
# Helpers
# ------------------------------
def detect_task(y: pd.Series) -> str:
    """Return 'binary', 'multiclass', or 'regression'."""
    if pd.api.types.is_numeric_dtype(y):
        # numeric but could be categorical if few unique values
        uniq = y.dropna().unique()
        if len(uniq) <= 10 and all(float(v).is_integer() for v in uniq):
            # treat small integer sets as classification
            if len(uniq) == 2:
                return "binary"
            elif len(uniq) > 2:
                return "multiclass"
        # otherwise regression
        return "regression"
    else:
        # non-numeric -> classification
        n_classes = y.dropna().nunique()
        return "binary" if n_classes == 2 else "multiclass"


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    cat_features = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category" or X[c].dtype == "bool"]
    num_features = [c for c in X.columns if c not in cat_features]
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features),
        ]
    )
    return pre, num_features, cat_features


def safe_train_test_split(
    X, y, test_size=0.2, random_state=42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, bool]:
    """Try stratified split; fall back if some class is too small."""
    vc = y.value_counts(dropna=False)
    multi_class = len(vc) > 1
    min_count = int(vc.min()) if len(vc) else 0

    can_stratify = (
        multi_class
        and min_count >= 2
        and (min_count * test_size) >= 1
        and (min_count * (1 - test_size)) >= 1
    )
    stratify_arg = y if can_stratify else None

    if not can_stratify and multi_class:
        st.warning(
            f"Some class has too few samples for a stratified split "
            f"(min class count = {min_count}). Falling back to non-stratified split."
        )
    elif not multi_class:
        st.warning("Only one class present in target. Metrics will be trivial; model cannot generalize.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    return X_train, X_test, y_train, y_test, bool(can_stratify)


def get_binary_labels(y: pd.Series, positive_label):
    """Map y to 0/1 with selected positive label -> 1."""
    return (y == positive_label).astype(int)


def prob_or_score(model, X, positive_index: Optional[int] = None) -> np.ndarray:
    """
    Return a 1D array of confidence for the positive class when possible.
    Falls back gracefully if predict_proba/decision_function are unavailable.
    """
    # predict_proba if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            idx = positive_index if positive_index is not None else 1
            idx = min(idx, proba.shape[1] - 1)
            return proba[:, idx]
        else:
            # single-column probability: use it
            return proba.ravel()

    # decision_function if available
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores.ravel()

    # fallback: use predicted labels as 0/1 score
    preds = model.predict(X)
    # try to map to 0/1
    try:
        preds = preds.astype(float)
    except Exception:
        preds = (preds == preds.max()).astype(float)
    return preds.ravel()


def confusion_matrix_figure(y_true, y_pred, labels) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[str(l) for l in labels],
            y=[str(l) for l in labels],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
    return fig


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="General ML App", layout="wide")

st.sidebar.header("User Info")
user_name = st.sidebar.text_input("Name", value="")
user_number = st.sidebar.number_input("Number", value=0, step=1)

st.title("📊 General ML App (Tabular)")
st.caption("Upload a CSV, explore EDA, train/evaluate a model, and make predictions.")

uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])

tabs = st.tabs(["1) Data", "2) EDA – Visual", "3) Train / Evaluate", "4) Prediction"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV to get started.")
    df = None

# Session state to store trained artifacts
if "trained_pipe" not in st.session_state:
    st.session_state.trained_pipe = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "task_type" not in st.session_state:
    st.session_state.task_type = None
if "classes_" not in st.session_state:
    st.session_state.classes_ = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None


# ------------------------------
# Tab 1: Data
# ------------------------------
with tabs[0]:
    st.subheader("Dataset Preview")
    if df is not None:
        st.write("Shape:", df.shape)
        st.dataframe(df.head(50))
        st.write("Missing values per column:")
        st.write(df.isna().sum())
        st.download_button(
            "Download current data as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="dataset.csv",
            mime="text/csv",
        )


# ------------------------------
# Tab 2: EDA – Visual
# ------------------------------
with tabs[1]:
    st.subheader("Exploratory Data Analysis")
    if df is None:
        st.info("Upload data in the **Data** tab first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            col_choice = st.selectbox("Pick a column for a plot", df.columns)
            if pd.api.types.is_numeric_dtype(df[col_choice]):
                fig = px.histogram(df, x=col_choice, nbins=30, title=f"Histogram: {col_choice}")
            else:
                vc = df[col_choice].value_counts().reset_index()
                vc.columns = [col_choice, "count"]
                fig = px.bar(vc.head(30), x=col_choice, y="count", title=f"Top categories: {col_choice}")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) >= 2:
                corr = df[num_cols].corr(numeric_only=True)
                fig2 = px.imshow(corr, text_auto=True, title="Correlation (numeric)")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for a correlation heatmap.")


# ------------------------------
# Tab 3: Train / Evaluate
# ------------------------------
with tabs[2]:
    st.subheader("Train / Evaluate")

    if df is None:
        st.info("Upload data in the **Data** tab first.")
    else:
        # Select target column
        target_col = st.selectbox("Select target column", df.columns, index=len(df.columns) - 1)
        y_raw = df[target_col]
        X_raw = df.drop(columns=[target_col])

        # Detect task
        task = detect_task(y_raw)
        st.write(f"Detected task: **{task}**")

        # If binary, allow choosing positive class
        pos_label = None
        positive_index = None
        if task == "binary":
            classes = sorted(y_raw.dropna().unique().tolist(), key=lambda x: str(x))
            pos_label = st.text_input(
                "Positive class label (exact match)", value=str(classes[-1] if classes else "")
            )
            # Try to coerce pos_label to original dtype
            if len(classes) > 0:
                try:
                    if isinstance(classes[0], (int, np.integer)):
                        pos_label = int(pos_label)
                    elif isinstance(classes[0], float):
                        pos_label = float(pos_label)
                except Exception:
                    pass

        # Split controls
        c1, c2 = st.columns(2)
        with c1:
            test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        with c2:
            random_state = st.number_input("Random state", 0, 10_000, 42, step=1)

        # Feature selection: auto all non-target
        X = X_raw.copy()
        y = y_raw.copy()

        # Build preprocessor
        pre, num_feats, cat_feats = build_preprocessor(X)
        st.write(f"Numeric features ({len(num_feats)}): {num_feats}")
        st.write(f"Categorical features ({len(cat_feats)}): {cat_feats}")

        # Prepare target for binary case
        if task == "binary":
            uniques = sorted(y.dropna().unique(), key=lambda x: str(x))
            if pos_label not in uniques:
                st.warning(
                    f"Selected positive label '{pos_label}' not in data classes {uniques}. "
                    f"Using default '{uniques[-1] if uniques else None}'."
                )
                pos_label = uniques[-1] if uniques else None

            if pos_label is not None:
                y_bin = get_binary_labels(y, pos_label)
            else:
                # Single class corner case
                y_bin = pd.Series(np.zeros(len(y), dtype=int), index=y.index)
            y_for_split = y_bin
        else:
            y_for_split = y

        # Split safely
        X_train, X_test, y_train, y_test, stratified = safe_train_test_split(
            X, y_for_split, test_size=test_size, random_state=random_state
        )

        # Choose model
        if task == "regression":
            model = RandomForestRegressor(
                n_estimators=200, random_state=random_state, n_jobs=-1
            )
        elif task == "binary":
            # Use RF for robust predict_proba
            model = RandomForestClassifier(
                n_estimators=300, random_state=random_state, n_jobs=-1, class_weight="balanced"
            )
        else:  # multiclass
            model = RandomForestClassifier(
                n_estimators=300, random_state=random_state, n_jobs=-1, class_weight="balanced"
            )

        pipe = Pipeline(steps=[("pre", pre), ("model", model)])

        if st.button("Train model"):
            pipe.fit(X_train, y_train)

            # Store for Prediction tab
            st.session_state.trained_pipe = pipe
            st.session_state.feature_cols = X.columns.tolist()
            st.session_state.task_type = task
            st.session_state.target_col = target_col

            # Evaluation
            y_pred = pipe.predict(X_test)

            if task == "regression":
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.success(f"MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {np.sqrt(mse):.4f} | R²: {r2:.4f}")
            elif task == "binary":
                # find index of positive class in model.classes_ for proba
                positive_index = None
                if hasattr(pipe.named_steps["model"], "classes_"):
                    classes_ = list(pipe.named_steps["model"].classes_)
                    st.session_state.classes_ = classes_
                    try:
                        # pos_label already coerced
                        positive_index = classes_.index(1) if set(y_train.unique()) <= {0,1} else classes_.index(pos_label)
                    except Exception:
                        positive_index = 1 if len(classes_) > 1 else 0

                y_score = prob_or_score(pipe, X_test, positive_index)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                st.write("### Metrics")
                st.write(f"Accuracy: **{acc:.4f}**  |  Precision: **{prec:.4f}**  |  Recall: **{rec:.4f}**  |  F1: **{f1:.4f}**")
                st.plotly_chart(confusion_matrix_figure(y_test, y_pred, labels=[0, 1]), use_container_width=True)

                if acc >= 0.85 and prec >= 0.85 and rec >= 0.85 and f1 >= 0.85:
                    st.success("✅ Meets the >85% requirement for Accuracy/Precision/Recall/F1.")
                else:
                    st.warning("⚠️ Metrics don’t all exceed 85%. Consider more data or different features/model.")
            else:
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                st.write("### Metrics (weighted)")
                st.write(f"Accuracy: **{acc:.4f}**  |  Precision: **{prec:.4f}**  |  Recall: **{rec:.4f}**  |  F1: **{f1:.4f}**")
                st.plotly_chart(confusion_matrix_figure(y_test, y_pred, labels=np.unique(y_test)), use_container_width=True)

            # Classification report (if classification)
            if task in ["binary", "multiclass"]:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                st.json(report)

            st.success("Training finished. Switch to the **Prediction** tab to try single predictions.")


# ------------------------------
# Tab 4: Prediction
# ------------------------------
with tabs[3]:
    st.subheader("Single Prediction")
    if st.session_state.trained_pipe is None:
        st.info("Train a model in the **Train / Evaluate** tab first.")
    else:
        feature_cols = st.session_state.feature_cols
        task = st.session_state.task_type

        st.write("Enter feature values:")
        inputs = {}
        # Dynamically build inputs
        for col in feature_cols:
            inputs[col] = st.text_input(col, value="")

        if st.button("Predict"):
            # Cast numeric columns to float when possible
            X_input = pd.DataFrame([inputs])
            # Try to infer dtypes from training data types (best effort)
            # Here we simply attempt numeric cast; if fails, keep as string -> handled by preprocessor
            for c in X_input.columns:
                try:
                    X_input[c] = pd.to_numeric(X_input[c])
                except Exception:
                    pass

            pipe = st.session_state.trained_pipe
            pred = pipe.predict(X_input)[0]

            if task in ["binary", "multiclass"]:
                # probability if available
                try:
                    proba = prob_or_score(pipe, X_input)
                    if proba.ndim == 0:
                        proba = np.array([proba])
                    st.write(f"Prediction: **{pred}**")
                    st.write(f"Confidence/Score: **{float(proba.ravel()[0]):.4f}**")
                except Exception:
                    st.write(f"Prediction: **{pred}**")
            else:
                st.write(f"Predicted value: **{float(pred):.4f}**")

