import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "mlapp/models/model.pkl"

def train_model(csv_file):
    df = pd.read_csv(csv_file)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    acc = model.score(X_test, y_test)
    return {"status": "Modelo entrenado", "accuracy": acc}

def predict_with_model(csv_file):
    df = pd.read_csv(csv_file)
    model = joblib.load(MODEL_PATH)
    preds = model.predict(df)
    df["prediction"] = preds
    return df
