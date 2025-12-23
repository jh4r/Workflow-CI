import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# 1. Load Data
df = pd.read_csv('telco_churn_clean.csv')

# Pisahkan Fitur & Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Setup MLflow
# Wajib arahkan ke localhost
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Eksperimen_Juhar_Model")

# 3. Training dengan Autolog
mlflow.autolog()

print("Memulai training...")
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Training Selesai. Akurasi: {acc:.4f}")
    print("Cek dashboard MLflow di http://127.0.0.1:5000")