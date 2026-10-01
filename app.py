import streamlit as st
import pandas as pd
import joblib

model = joblib.load("cloud_anomaly_model.pkl")
df = pd.read_csv("test.csv")

feature_columns = [
    "cpu-N1", "cpu-N2", "cpu-N3", "cpu-N4", "cpu-N6", "cpu-N7", "cpu-N8",
    "mem_gb-N1", "mem_gb-N2", "mem_gb-N3", "mem_gb-N4",
    "mem_gb-N6", "mem_gb-N7", "mem_gb-N8",
    "disk_io-N1", "disk_io-N2", "disk_io-N3", "disk_io-N4",
    "disk_io-N6", "disk_io-N7", "disk_io-N8",
    "net_in-N1", "net_in-N2", "net_in-N3", "net_in-N4",
    "net_in-N6", "net_in-N7", "net_in-N8",
    "cluster_pending_cluster",
    "cluster_api_latency_cluster",
    "cluster_restarts_cluster"
]

st.title("Cloud Service Resource Anomaly Detection")

st.write("Select a data sample from the dataset:")

sample_number = st.number_input(
    "Sample Number",
    min_value=1,
    max_value=len(df),
    value=1,
    step=1
)

if st.button("Predict"):

    row = df.iloc[sample_number - 1]

    input_data = pd.DataFrame(
        [row[feature_columns].values],
        columns=feature_columns
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write("Anomaly Risk:", round(probability * 100, 2), "%")
    st.progress(float(probability))

    if prediction == 1:
        st.error("⚠️ Anomaly Detected")
    else:
        st.success("✅ Normal")

    st.write("Actual Label:", int(row["label"]))