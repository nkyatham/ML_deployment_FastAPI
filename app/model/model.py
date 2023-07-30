import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from pathlib import Path


__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

model = joblib.load(f"{BASE_DIR}/XGBoost-{__version__}.joblib")
pipeline = joblib.load(f"{BASE_DIR}/transformation_pipeline-{__version__}.joblib")


def model_predict(payload):
    data = np.array(list(payload.values()))
    binarizer = Pipeline([
        ('binarizer', pipeline.named_steps['binarizer'])
    ])
    data[40] = int(binarizer.transform([data[40]]))
    infer_data_pipeline = Pipeline([
        ('scaler', pipeline.named_steps['scaler']),
        ('imputer', pipeline.named_steps['imputer'])
    ])
    datapoint = infer_data_pipeline.transform(data[1:].reshape(1,-1))
    pred_proba = model.predict(datapoint)
    return pred_proba
