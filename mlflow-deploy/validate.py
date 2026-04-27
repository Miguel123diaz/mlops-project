import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import sys
import os
import mlflow

# Configurar MLflow para guardar en carpeta local
mlflow.set_tracking_uri("file:./mlflow-deploy/mlruns")
mlflow.set_experiment("drug-classification")

# Parámetro de umbral
THRESHOLD = 5000.0

# --- Cargar dataset ---
print("--- Debug: Cargando dataset load_diabetes ---")
X, y = load_diabetes(return_X_y=True, as_frame=True)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"--- Debug: Dimensiones de X_test: {X_test.shape} ---")

# --- Cargar modelo previamente entrenado ---
model_filename = "model.pkl"
model_path = os.path.abspath(os.path.join(os.getcwd(), model_filename))
print(f"--- Debug: Intentando cargar modelo desde: {model_path} ---")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"--- ERROR: No se encontró el archivo del modelo en '{model_path}'. ---")
    print(f"--- Debug: Archivos en {os.getcwd()}: ---")
    try:
        print(os.listdir(os.getcwd()))
    except Exception as list_err:
        print(f"(No se pudo listar el directorio: {list_err})")
    sys.exit(1)

# --- MLflow Run para validación ---
with mlflow.start_run(run_name="validation"):
    print("--- Debug: Realizando predicciones ---")
    try:
        y_pred = model.predict(X_test)
    except ValueError as pred_err:
        print(f"--- ERROR durante la predicción: {pred_err} ---")
        print(f"Modelo esperaba {model.n_features_in_} features.")
        print(f"X_test tiene {X_test.shape[1]} features.")
        sys.exit(1)

    mse = mean_squared_error(y_test, y_pred)
    print(f"🔍 MSE del modelo: {mse:.4f} (umbral: {THRESHOLD})")

    # Loguear métrica en MLflow
    mlflow.log_metric("mse", mse)

    # Guardar resultados en archivo
    os.makedirs("./Results", exist_ok=True)
    with open("./Results/validation.txt", "w") as f:
        f.write(f"MSE = {mse:.4f}\nUmbral = {THRESHOLD}")

    # Subir archivo como artifact
    mlflow.log_artifact("./Results/validation.txt")

    # Validación
    if mse <= THRESHOLD:
        print("✅ El modelo cumple los criterios de calidad.")
        sys.exit(0)
    else:
        print("❌ El modelo no cumple el umbral. Deteniendo pipeline.")
        sys.exit(1)
