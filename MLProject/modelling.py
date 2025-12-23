#pakai file yang modelling_tuning.py

import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import shutil
import os

# 1. Load Data
df = pd.read_csv("telco_churn_clean.csv")

# Pisahkan Fitur dan Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Setup MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Eksperimen_Juhar_SkilledV4") 

# 4. Tuning
est = RandomForestClassifier(random_state=42)
params = {
    "n_estimators": [50, 100],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5]
}

search = RandomizedSearchCV(est, params, n_iter=5, cv=3, n_jobs=-1, verbose=1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
best_params = search.best_params_

# 5. Loggging ke MLflow
with mlflow.start_run(run_name="Skilled_TuningV4"):
    mlflow.log_params(best_params)

    y_pred = best_model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    
    # Simpan dulu modelnya di folder lokal laptop
    local_model_path = "model_temp"
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path) # Hapus kalo bekas sisa yg lama
    
    # Ini akan membuat folder 'model_temp' yg isinya MLmodel, conda.yaml, dll
    mlflow.sklearn.save_model(best_model, local_model_path)
    
    # Upload folder itu ke MLflow sebagai artifact bernama "model"
    print("...Mengupload folder model ke MLflow...")
    mlflow.log_artifacts(local_model_path, artifact_path="model")
    
    # Hapus folder temp biar bersih
    shutil.rmtree(local_model_path)
    
    # Simpan Gambar
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.savefig("training_confusion_matrix.png")
    mlflow.log_artifact("training_confusion_matrix.png")

print("Training Selesai! Cek MLflow sekarang")
