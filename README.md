# Cloud Service Resource Anomaly Classification

## Objective
To identify unusual resource usage in a simulated cloud-service environment.

## Inputs
- CPU Usage
- Memory Usage
- Disk I/O
- Network Throughput
- API Latency
- Pending Clusters
- Cluster Restarts

## Machine Learning Model
Decision Tree Classifier

## Evaluation
The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Application
A Streamlit web application is used to predict whether the resource usage is Normal or an Anomaly.

## Technologies
- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit