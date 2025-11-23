import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import (
    KFold, LeaveOneOut, cross_val_predict, train_test_split
)
from sklearn.metrics import (
    accuracy_score, log_loss, confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")
import joblib
import os

# -------------------------------------------------------------------
# APP SETUP
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Big Data Analytics",
    page_icon="📘",
    layout="wide"
)

# -------------------------------------------------------------------
# CUSTOM CSS – FULL UI REDESIGN
# -------------------------------------------------------------------
st.markdown("""
<style>
/* Global */
* { font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { background-color: #0f172a; }

/* Main background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0f172a 0, #020617 40%, #000 100%);
    color: #e5e7eb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #020617 50%, #030712 100%);
    border-right: 1px solid rgba(148,163,184,0.2);
}
[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Header title */
.app-header {
    padding: 1.25rem 1.5rem 0.75rem 1.5rem;
    border-radius: 1.25rem;
    background: linear-gradient(120deg, #0ea5e9 0%, #4f46e5 45%, #22c55e 100%);
    color: white;
    box-shadow: 0 20px 40px rgba(15,23,42,0.5);
    margin-bottom: 1.5rem;
}
.app-header h1 {
    margin-bottom: 0.3rem;
}
.app-header p {
    margin-top: 0;
    opacity: 0.9;
}

/* Section card */
.card {
    background: rgba(15,23,42,0.86);
    border-radius: 1.25rem;
    padding: 1.25rem 1.5rem;
    border: 1px solid rgba(148,163,184,0.25);
    box-shadow: 0 18px 35px rgba(15,23,42,0.7);
    margin-bottom: 1.2rem;
}

/* Subsection header */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.section-title span.icon {
    font-size: 1.3rem;
}

/* Metric cards */
.metric-card {
    background: radial-gradient(circle at top left, rgba(56,189,248,0.22), rgba(15,23,42,0.95));
    border-radius: 1rem;
    padding: 0.9rem 1rem;
    border: 1px solid rgba(59,130,246,0.4);
    box-shadow: 0 12px 25px rgba(15,23,42,0.9);
}
.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #9ca3af;
    margin-bottom: 0.2rem;
}
.metric-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e5e7eb;
}

/* Tabs */
.stTabs [role="tablist"] {
    gap: 0.5rem;
}
.stTabs [role="tab"] {
    padding: 0.7rem 1.3rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.4);
    background: rgba(15,23,42,0.7);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(120deg, #0ea5e9, #6366f1);
    color: white !important;
    border-color: transparent;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 0.9rem;
    overflow: hidden;
    border: 1px solid rgba(55,65,81,0.9);
}

/* Sliders */
span[role="slider"] {
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
}

/* Buttons */
.stButton>button {
    border-radius: 999px;
    padding: 0.5rem 1.4rem;
    background: linear-gradient(120deg, #22c55e, #16a34a);
    color: white;
    border: none;
    font-weight: 600;
}
.stButton>button:hover {
    filter: brightness(1.1);
}

/* Divider text */
hr {
    border-color: rgba(55,65,81,0.8);
}

/* Captions */
small, .caption {
    color: #9ca3af !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📘 Big Data Analytics - Laboratory 2")
    st.markdown("**Topic:** Resampling Techniques & Performance Metrics")

    st.markdown("---")
    st.markdown("#### 🧪 Tasks")
    st.markdown("- Part 1: **Classification**\n- Part 2: **Regression**")

    st.markdown("---")
    st.markdown("#### 📂 Datasets Used")
    st.markdown("- Diabetes\n- Climate Change Impacts")

    st.markdown("---")
    st.markdown("#### 🔍 Quick Legend")
    st.markdown("- **K-Fold CV:** Balanced evaluation\n- **LOOCV:** Thorough but slower\n- **Repeated Splits:** More stable regression metrics")

# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
st.markdown("""
<div class="app-header">
  <h1> Big Data Analytics - Laboratory 2 </h1>
  <p>Utilizing <b>Resampling Techniques</b> and <b>Performance Metrics</b> for Classification & Regression</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Part 1:** Classification – Health-related dataset (Diabetes)  
**Part 2:** Regression – Environment-related dataset (Climate impacts)
""")

# Load datasets
diabetes = pd.read_csv("diabetes.csv")
climate = pd.read_csv("realistic_climate_change_impacts.csv")

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def evaluate_classification(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob[:, 1])
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, output_dict=True)
    return acc, ll, auc, cm, cr

def plot_confusion(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig

def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    plt.tight_layout()
    return fig

def metric_card(label, value, fmt=".4f"):
    try:
        val_str = format(value, fmt)
    except Exception:
        val_str = str(value)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val_str}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------------
# TAB SETUP
# -------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Overview & Preprocessing",
    "🩺 Part 1 – Classification",
    "🌿 Part 2 – Regression"
])

# -------------------------------------------------------------------
# TAB 1: Data Overview & Preprocessing
# -------------------------------------------------------------------
with tab1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-title">
            <span class="icon">📌</span> <span>Lab Exercise Overview</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This lab focuses on exploring **resampling techniques** and **performance evaluation metrics**
        using two types of machine learning tasks:

        - 🩺 **Part 1 – Classification:** Predict diabetes using Logistic Regression  
        - 🌿 **Part 2 – Regression:** Predict environmental impact using Linear Regression
        """)

        st.markdown("</div>", unsafe_allow_html=True)

    # ============================
    # DIABETES – CLASSIFICATION
    # ============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-title">
        <span class="icon">🩺</span> <span>Part 1: Diabetes Dataset (Classification)</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Dataset Description & Features", expanded=True):
        st.markdown("""
        The **Diabetes Dataset** is used to predict whether a patient is diabetic.

        **Key Features:**
        - `Pregnancies`
        - `Glucose`
        - `BloodPressure`
        - `SkinThickness`
        - `Insulin`
        - `BMI`
        - `DiabetesPedigreeFunction`
        - `Age`
        - `Outcome` – 1 = Diabetic, 0 = Non-diabetic
        """)

    st.markdown("#### 🔍 Step 1 – First 5 Rows")
    st.dataframe(diabetes.head())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧬 Data Types")
        st.dataframe(diabetes.dtypes.astype(str).rename("Type").to_frame())
    with col2:
        st.markdown("#### ❓ Missing Values")
        st.dataframe(diabetes.isnull().sum().rename("Missing Values").to_frame())

    st.markdown("""
    Some health-related columns may contain **invalid zeros** (e.g., Glucose = 0) which are not realistic.
    We treat these as *missing-like values* and replace them with the column median.
    """)

    cols_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in cols_zero:
        diabetes.loc[diabetes[c] == 0, c] = diabetes[diabetes[c] != 0][c].median()
    st.success("Replaced invalid zero values with median values for affected health features.")

    st.markdown("#### ✅ Post-Preprocessing Preview")
    st.dataframe(diabetes.head())

    st.markdown("#### 🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(diabetes.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.caption("Stronger correlations with `Outcome` indicate more predictive features (e.g., `Glucose`, `BMI`).")

    st.markdown('</div>', unsafe_allow_html=True)

    # ============================
    # CLIMATE – REGRESSION
    # ============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-title">
        <span class="icon">🌿</span> <span>Part 2: Climate Change Impacts Dataset (Regression)</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Dataset Description & Features", expanded=True):
        st.markdown("""
        This dataset includes climate-related indicators such as **CO₂ levels**, **temperature anomalies**,  
        and **economic impact** of extreme weather events.

        **Example Features:**
        - `Date`
        - `Country`
        - `TemperatureAnomaly_C`
        - `CO2Level_ppm`
        - `ExtremeWeatherEvent`
        - `EconomicImpact_USD`
        - `PopulationAffected`
        """)

    st.markdown("#### 🔍 Step 1 – First 5 Rows")
    st.dataframe(climate.head())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧬 Data Types")
        st.dataframe(climate.dtypes.astype(str).rename("Type").to_frame())
    with col2:
        st.markdown("#### ❓ Missing Values")
        st.dataframe(climate.isnull().sum().rename("Missing Values").to_frame())

    st.markdown("""
    For **Linear Regression**, we need **numeric features only**.
    Non-numeric columns will be dropped, and missing numeric values will be replaced with medians.
    """)

    non_numeric_cols = climate.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        st.warning(f"Removed non-numeric columns: {non_numeric_cols}")
        climate = climate.drop(columns=non_numeric_cols)

    climate = climate.fillna(climate.median())
    st.success("Cleaned climate dataset: removed text columns and replaced missing numeric values with medians.")

    st.markdown("#### ✅ Post-Cleaning Preview")
    st.dataframe(climate.head())

    if not climate.empty:
        st.markdown("#### 🔗 Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.heatmap(climate.corr(), annot=True, cmap="YlOrRd", ax=ax2)
        st.pyplot(fig2)
        st.caption("Look for strong correlations between CO₂, temperature anomalies, and impact variables.")

    st.markdown("""
    **Preprocessing Summary:**
    - Replaced invalid / missing values with medians  
    - Removed non-numeric columns for regression  
    - Verified correlation structure for both tasks
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# TAB 2: Classification – Logistic Regression (Diabetes)
# -------------------------------------------------------------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-title">
        <span class="icon">🩺</span> <span>Part 1 – Classification Models (Diabetes Dataset)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Goal:** Predict whether a patient has diabetes using **Logistic Regression** with two resampling strategies:

    - **Model A:** K-Fold Cross-Validation  
    - **Model B:** Leave-One-Out Cross-Validation (LOOCV)
    """)

    st.markdown('</div>', unsafe_allow_html=True)

    X = diabetes.drop("Outcome", axis=1)
    y = diabetes["Outcome"]
    model_base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))

    colA, colB = st.columns(2)

    # MODEL A – K-FOLD
    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-title">
            <span class="icon">🧩</span> <span>Model A – K-Fold Cross Validation</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        K-Fold CV splits the data into **K parts** and rotates which part is used as test data.
        This gives a **balanced estimate** of performance.
        """)

        k = st.slider("Select number of folds (K)", 3, 15, 10, key="kfold_slider")
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        y_prob_A = cross_val_predict(model_base, X, y, cv=kf, method="predict_proba")
        y_pred_A = np.argmax(y_prob_A, axis=1)

        acc_A = accuracy_score(y, y_pred_A)
        ll_A = log_loss(y, y_prob_A)
        auc_A = roc_auc_score(y, y_prob_A[:, 1])
        cm_A = confusion_matrix(y, y_pred_A)
        cr_A = classification_report(y, y_pred_A, output_dict=True)

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            metric_card("Accuracy", acc_A)
        with mcol2:
            metric_card("Log Loss", ll_A)
        with mcol3:
            metric_card("AUC", auc_A)

        st.markdown("##### Confusion Matrix")
        figA1 = plot_confusion(cm_A)
        st.pyplot(figA1)

        st.markdown("##### ROC Curve")
        figA2 = plot_roc(y, y_prob_A[:, 1])
        st.pyplot(figA2)

        with st.expander("Show Classification Report"):
            st.dataframe(pd.DataFrame(cr_A).transpose())

        st.markdown("""
        - Higher **Accuracy** = more correct predictions  
        - Lower **Log Loss** = better calibrated probabilities  
        - Higher **AUC** = better separation between diabetic and non-diabetic patients
        """)

        st.markdown('</div>', unsafe_allow_html=True)

    # MODEL B – LOOCV
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-title">
            <span class="icon">🔁</span> <span>Model B – Leave-One-Out CV (LOOCV)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        LOOCV is an extreme case of K-Fold where **K = number of samples**.  
        Each sample is tested once while the rest are used for training.
        """)

        loo = LeaveOneOut()
        y_prob_B = cross_val_predict(model_base, X, y, cv=loo, method="predict_proba")
        y_pred_B = np.argmax(y_prob_B, axis=1)

        acc_B = accuracy_score(y, y_pred_B)
        ll_B = log_loss(y, y_prob_B)
        auc_B = roc_auc_score(y, y_prob_B[:, 1])
        cm_B = confusion_matrix(y, y_pred_B)
        cr_B = classification_report(y, y_pred_B, output_dict=True)

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            metric_card("Accuracy", acc_B)
        with mcol2:
            metric_card("Log Loss", ll_B)
        with mcol3:
            metric_card("AUC", auc_B)

        st.markdown("##### Confusion Matrix")
        figB1 = plot_confusion(cm_B)
        st.pyplot(figB1)

        st.markdown("##### ROC Curve")
        figB2 = plot_roc(y, y_prob_B[:, 1])
        st.pyplot(figB2)

        with st.expander("Show Classification Report"):
            st.dataframe(pd.DataFrame(cr_B).transpose())

        st.markdown("""
        LOOCV uses **every sample** as a test case once.  
        It is thorough but more computationally expensive and sometimes higher variance.
        """)

        st.markdown('</div>', unsafe_allow_html=True)

    # COMPARISON + INTERACTIVE PREDICTION
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-title">
        <span class="icon">📊</span> <span>Model Comparison & Custom Prediction</span>
    </div>
    """, unsafe_allow_html=True)

    comp_df = pd.DataFrame({
        "Metric": ["Accuracy", "Log Loss", "AUC"],
        "Model A (K-Fold)": [acc_A, ll_A, auc_A],
        "Model B (LOOCV)": [acc_B, ll_B, auc_B],
    })
    st.markdown("#### 🧮 Side-by-Side Metrics")
    st.dataframe(comp_df)

    st.markdown("""
    - **Model A** is usually preferred: stable and efficient  
    - **Model B** gives a very detailed estimate but is slower
    """)

    st.markdown("#### 🔮 Try a Custom Prediction (Using Model A)")
    model_final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    model_final.fit(X, y)
    # Save the selected model (Model A: K-Fold final model) to disk so it can be downloaded
    try:
        joblib.dump(model_final, "model_kfold.pkl")
    except Exception:
        # If saving fails for any reason (e.g., permission), continue without crashing the app
        st.warning("Could not save trained model to disk.")

    # Provide a download button for the trained model
    if os.path.exists("model_kfold.pkl"):
        with open("model_kfold.pkl", "rb") as f:
            model_bytes = f.read()
        st.download_button(
            "Download Trained Model (K-Fold)",
            data=model_bytes,
            file_name="model_kfold.pkl",
            mime="application/octet-stream"
        )

    # Prefer loading saved model for predictions if available
    if os.path.exists("model_kfold.pkl"):
        try:
            clf_for_pred = joblib.load("model_kfold.pkl")
        except Exception:
            clf_for_pred = model_final
    else:
        clf_for_pred = model_final
    input_vals = {}
    cols_pred = st.columns(3)
    for i, c in enumerate(X.columns):
        with cols_pred[i % 3]:
            input_vals[c] = st.number_input(
                f"{c}", float(X[c].min()), float(X[c].max()), float(X[c].mean())
            )

    if st.button("Predict Diabetes Outcome"):
        df_in = pd.DataFrame([input_vals])
        prob = clf_for_pred.predict_proba(df_in)[0][1]
        pred = int(prob >= 0.5)
        st.success(f"Predicted Outcome: {'Diabetic (1)' if pred == 1 else 'Non-Diabetic (0)'}")
        st.write(f"**Probability (class = 1): {prob:.4f}**")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# TAB 3: Regression – Linear Regression (Climate)
# -------------------------------------------------------------------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-title">
        <span class="icon">🌿</span> <span>Part 2 – Regression Task using Linear Regression</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Goal:** Predict an environmental impact variable (e.g., economic loss, affected population) using **Linear Regression**.

    We compare:
    - **Model A:** Single Train/Test Split  
    - **Model B:** Repeated Random Train/Test Splits
    """)

    st.markdown('</div>', unsafe_allow_html=True)

    X2 = climate.drop(climate.columns[-1], axis=1)
    y2 = climate[climate.columns[-1]]
    model2 = LinearRegression()

    colA, colB = st.columns(2)

    # MODEL A – Train/Test Split
    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-title">
            <span class="icon">🧪</span> <span>Model A – Single Train/Test Split</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        The dataset is split into **training** and **testing** sets once.
        Simple and fast, but results can depend heavily on the random split.
        """)

        test_size = st.slider("Test Size (Model A)", 0.1, 0.5, 0.2, key="testA")
        X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=test_size, random_state=42)
        model2.fit(X_train, y_train)
        preds_A = model2.predict(X_test)

        mse_A = mean_squared_error(y_test, preds_A)
        mae_A = mean_absolute_error(y_test, preds_A)
        r2_A = r2_score(y_test, preds_A)

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            metric_card("MSE", mse_A)
        with mcol2:
            metric_card("MAE", mae_A)
        with mcol3:
            metric_card("R²", r2_A)

        st.markdown("##### Actual vs Predicted")
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, preds_A, alpha=0.7)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title("Actual vs Predicted (Model A)")
        st.pyplot(fig1)

        residuals_A = y_test - preds_A
        st.markdown("##### Residuals Plot")
        fig2, ax2 = plt.subplots()
        ax2.scatter(preds_A, residuals_A, alpha=0.6)
        ax2.axhline(0, color="red", linestyle="--")
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals vs Predicted (Model A)")
        st.pyplot(fig2)

        st.markdown("""
        - **MSE / MAE** indicate typical error size  
        - **R²** shows how much variance is explained by the model  
        - Random residual scatter around 0 suggests a reasonable linear fit
        """)

        st.markdown('</div>', unsafe_allow_html=True)

    # MODEL B – Repeated Random Splits
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-title">
            <span class="icon">🔁</span> <span>Model B – Repeated Random Train/Test Splits</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        Here we perform **multiple random splits**, train, evaluate, and then **average** the results.
        This reduces the effect of a single lucky/unlucky split.
        """)

        n_repeats = st.slider("Number of Random Splits", 2, 10, 5, key="repeatB")
        mse_list, mae_list, r2_list = [], [], []
        all_preds, all_actuals = [], []

        for seed in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=seed)
            model2.fit(X_train, y_train)
            preds = model2.predict(X_test)
            mse_list.append(mean_squared_error(y_test, preds))
            mae_list.append(mean_absolute_error(y_test, preds))
            r2_list.append(r2_score(y_test, preds))
            all_preds.extend(preds)
            all_actuals.extend(y_test)

        mse_B = np.mean(mse_list)
        mae_B = np.mean(mae_list)
        r2_B = np.mean(r2_list)

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            metric_card("Avg MSE", mse_B)
        with mcol2:
            metric_card("Avg MAE", mae_B)
        with mcol3:
            metric_card("Avg R²", r2_B)

        st.markdown("##### Actual vs Predicted (All Splits)")
        fig3, ax3 = plt.subplots()
        ax3.scatter(all_actuals, all_preds, alpha=0.6)
        ax3.plot([min(all_actuals), max(all_actuals)],
                 [min(all_actuals), max(all_actuals)], "r--")
        ax3.set_xlabel("Actual Values")
        ax3.set_ylabel("Predicted Values")
        ax3.set_title("Actual vs Predicted (Model B – Repeated Splits)")
        st.pyplot(fig3)

        st.markdown("##### Trend Comparison")
        fig4, ax4 = plt.subplots()
        ax4.plot(sorted(all_actuals), sorted(all_preds))
        ax4.set_xlabel("Actual (sorted)")
        ax4.set_ylabel("Predicted (sorted)")
        ax4.set_title("Trend Comparison – Model B")
        st.pyplot(fig4)

        st.markdown("""
        Averaging over multiple splits gives **more stable** estimates of MSE, MAE and R², making
        Model B more reliable in many practical scenarios.
        """)

        st.markdown('</div>', unsafe_allow_html=True)

    # COMPARISON + PREDICTION
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-title">
        <span class="icon">📊</span> <span>Model Comparison & Custom Prediction</span>
    </div>
    """, unsafe_allow_html=True)

    comparison_df = pd.DataFrame({
        "Model": ["Model A (Train/Test Split)", "Model B (Repeated Splits)"],
        "MSE": [mse_A, mse_B],
        "MAE": [mae_A, mae_B],
        "R²": [r2_A, r2_B]
    })
    st.markdown("#### 🧮 Side-by-Side Metrics")
    st.dataframe(comparison_df)

    st.markdown("""
    - Lower **MSE / MAE** = better average prediction accuracy  
    - Higher **R²** = better overall fit  
    - **Model B** is usually more stable because it averages over several random partitions.
    """)

    st.markdown("#### 🔮 Custom Prediction (Using Model B)")
    # Train a final regression model on the full dataset for deployment/download
    try:
        model2.fit(X2, y2)
        # Save the trained regression model
        try:
            joblib.dump(model2, "model_reg.pkl")
        except Exception:
            st.warning("Could not save regression model to disk.")
    except Exception:
        # If training fails for any reason, continue and rely on later fit
        st.warning("Final training of regression model failed; interactive prediction may retrain on demand.")

    # Provide download button for the regression model if available
    if os.path.exists("model_reg.pkl"):
        with open("model_reg.pkl", "rb") as f:
            reg_bytes = f.read()
        st.download_button(
            "Download Trained Regression Model",
            data=reg_bytes,
            file_name="model_reg.pkl",
            mime="application/octet-stream"
        )

    # Prefer loading saved model for predictions
    if os.path.exists("model_reg.pkl"):
        try:
            reg_for_pred = joblib.load("model_reg.pkl")
        except Exception:
            reg_for_pred = model2
    else:
        reg_for_pred = model2

    user_input = {}
    for col in X2.columns:
        user_input[col] = st.number_input(
            f"{col}", float(X2[col].min()), float(X2[col].max()), float(X2[col].mean())
        )

    if st.button("Predict Using Selected Model (Model B)"):
        input_df = pd.DataFrame([user_input])
        # Use the saved/loaded final model for prediction (avoid retraining here)
        try:
            prediction = reg_for_pred.predict(input_df)[0]
        except Exception:
            # Fallback: retrain on full data then predict
            model2.fit(X2, y2)
            prediction = model2.predict(input_df)[0]
        st.success(f"Predicted Value: {prediction:.4f}")

    st.markdown('</div>', unsafe_allow_html=True)
