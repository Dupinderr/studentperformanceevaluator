import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------
# App Configuration
# ------------------
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("📊 Student Performance Prediction App")

# ------------------
# Load Default Dataset
# ------------------
def load_data():
    return pd.read_csv("StudentsPerformance_3_.csv")

@st.cache_data
def preprocess_data(df):
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    return df

# ------------------
# Sidebar - File Upload
# ------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file is not None:
        st.success("✅ Custom dataset uploaded successfully")

        # Store uploaded file temporarily in session_state
        st.session_state["uploaded_df"] = pd.read_csv(uploaded_file)

        # Button to refresh & apply new dataset
        if st.button("🔄 Use New Dataset"):
            st.session_state["active_df"] = st.session_state["uploaded_df"]
            st.rerun()
    else:
        st.info("ℹ️ Using default dataset (StudentsPerformance_3_.csv)")
        if "active_df" not in st.session_state:
            st.session_state["active_df"] = load_data()

# ------------------
# Use Active Dataset
# ------------------
df = st.session_state["active_df"]

# ------------------
# Show Dataset
# ------------------
st.subheader("🔎 Dataset Preview")
st.dataframe(df.head())

# ------------------
# Preprocess and Train Model
# ------------------
df_processed = preprocess_data(df)

# ✅ Restrict target columns only to math/reading/writing scores
possible_targets = ["math score", "reading score", "writing score"]
target_cols = [col for col in df_processed.columns if col.lower() in [t.lower() for t in possible_targets]]

if not target_cols:
    st.error("❌ No valid score columns (math/reading/writing) found in this dataset.")
    st.stop()

X = df_processed.drop(columns=target_cols)
y = df_processed[target_cols]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# ------------------
# User Input for Prediction
# ------------------
st.subheader("🎯 Predict Student Performance")

input_data = {}
cols = st.columns(min(3, len(df.columns)))

for i, col in enumerate([c for c in df.columns if c not in target_cols]):
    with cols[i % len(cols)]:
        if df[col].dtype == "object":
            input_data[col] = st.selectbox(f"{col}", df[col].unique())
        else:
            input_data[col] = st.number_input(
                f"{col}", 
                float(df[col].min()), 
                float(df[col].max()), 
                float(df[col].mean())
            )

# Convert input into DataFrame
input_df = pd.DataFrame([input_data])
input_processed = preprocess_data(input_df)

# Align columns with training data
input_processed = input_processed.reindex(columns=X.columns, fill_value=0)

if st.button("Predict Performance"):
    prediction = model.predict(input_processed)
    st.success("✅ Prediction Completed")
    st.write("### Predicted Scores:")
    for i, col in enumerate(target_cols):
        st.write(f"📘 {col}: {prediction[0][i]:.1f}")

# ------------------
# Visualizations
# ------------------
st.subheader("📊 Data Visualizations")

viz_choice = st.selectbox("Choose a visualization", ["Correlation Heatmap", "Score Distributions"])

if viz_choice == "Correlation Heatmap":
    plt.figure(figsize=(8,6))
    sns.heatmap(df_processed.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)
elif viz_choice == "Score Distributions":
    fig, ax = plt.subplots(1, len(target_cols), figsize=(5*len(target_cols), 5))
    if len(target_cols) == 1:
        ax = [ax]  # ensure iterable
    for i, col in enumerate(target_cols):
        sns.histplot(df[col], bins=20, ax=ax[i])
        ax[i].set_title(f"{col} Distribution")
    st.pyplot(fig)
