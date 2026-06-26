# modelling.py
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_modelling():
    print("--- Memulai Pelatihan Otomatis via GitHub Actions CI ---")
    
    # Ambil URL tracking dari environment variable secara dinamis
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[INFO] Tracking URI disetel ke: {tracking_uri}")
    
    mlflow.set_experiment("Credit_Card_Basic_Experiment")
    
    # Path dinamis membaca folder dataset di dalam lingkungan MLProject
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "namadataset_preprocessing")
    
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    with mlflow.start_run(run_name="CI_Automated_Random_Forest"):
        n_estimators = 100
        max_depth = 5
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        
        # Log model ke MLflow Artifacts
        mlflow.sklearn.log_model(model, "baseline_model")
        
        print(f"[SUKSES] Training selesai. Akurasi CI: {acc:.4f}")

if __name__ == "__main__":
    run_modelling()