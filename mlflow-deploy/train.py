import pandas as pd
import skops.io as sio
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import os
import joblib

# Crear carpetas de resultados si no existen
os.makedirs("./Results", exist_ok=True)
os.makedirs("./Model", exist_ok=True)

# Configurar MLflow para guardar runs en carpeta local
mlflow.set_tracking_uri("file:./mlflow-deploy/mlruns")
mlflow.set_experiment("drug-classification")

## Loading the Data
drug_df = pd.read_csv("Data/drug.csv")
drug_df = drug_df.sample(frac=1)

## Train Test Split
X = drug_df.drop("Drug", axis=1).values
y = drug_df.Drug.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

## Pipeline
cat_col = [1, 2, 3]
num_col = [0, 4]

transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=10, random_state=125)),
    ]
)

## MLflow Run
with mlflow.start_run(run_name="training"):
    # Training
    pipe.fit(X_train, y_train)

    # Model Evaluation
    predictions = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

    # Log parameters and metrics in MLflow
    mlflow.log_param("n_estimators", 10)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log model in MLflow
    mlflow.sklearn.log_model(pipe, "model")

    ## Confusion Matrix Plot
    cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
    disp.plot()
    plt.savefig("./Results/model_results.png", dpi=120)

    # Log confusion matrix image in MLflow
    mlflow.log_artifact("./Results/model_results.png")

    ## Write metrics to file
    with open("./Results/metrics.txt", "w") as outfile:
        outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}")

    # Log metrics file in MLflow
    mlflow.log_artifact("./Results/metrics.txt")

    ## Saving the model file locally
    sio.dump(pipe, "./Model/drug_pipeline.skops")
    mlflow.log_artifact("./Model/drug_pipeline.skops")

    # Guardar también como model.pkl para validación
    joblib.dump(pipe, "model.pkl")
    mlflow.log_artifact("model.pkl")
