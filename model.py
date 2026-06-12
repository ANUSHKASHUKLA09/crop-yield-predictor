import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from xgboost import XGBRegressor
    USE_XGB = True
except ImportError:
    USE_XGB = False

from data_processor import generate_training_data

CROP_MAP   = {"Wheat": 0, "Rice": 1, "Sugarcane": 2, "Mustard": 3}
SEASON_MAP = {"rabi": 0, "kharif": 1, "zaid": 2}
FEATURE_COLS = ["ndvi", "rainfall", "temperature", "soil_quality", "crop", "season"]
MODEL_PATH = "crop_yield_model.pkl"


def train_model():
    """Train the yield prediction model and save it to disk."""
    print("🌱 Training crop yield model …")
    df = generate_training_data(n_samples=3000)

    X = df[FEATURE_COLS].values
    y = df["yield"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if USE_XGB:
        base_model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    else:
        base_model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", base_model),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"✅ Model trained — R² = {r2:.3f}, MAE = {mae:.3f} t/ha")

    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def predict_yield(model, features: dict) -> float:
    """
    Predict yield given a features dict with keys:
        ndvi, rainfall, temperature, soil_quality, crop (str), season (str)
    Returns yield in t/ha (float).
    """
    crop_enc   = CROP_MAP.get(features.get("crop", "Wheat"), 0)
    season_enc = SEASON_MAP.get(features.get("season", "rabi"), 0)

    X = np.array([[
        features["ndvi"],
        features["rainfall"],
        features["temperature"],
        features["soil_quality"],
        float(crop_enc),
        float(season_enc),
    ]])

    prediction = model.predict(X)[0]
    return round(float(np.clip(prediction, 0.5, 8.0)), 2)


def get_feature_importance(model) -> dict:
    """Return feature importance scores from the trained model."""
    try:
        importances = model.named_steps["model"].feature_importances_
        return dict(zip(FEATURE_COLS, importances.tolist()))
    except Exception:
        return {}


if __name__ == "__main__":
    m = train_model()
    sample = {
        "ndvi": 0.6,
        "rainfall": 800,
        "temperature": 26,
        "soil_quality": 7,
        "crop": "Wheat",
        "season": "rabi",
    }
    print(f"Sample prediction: {predict_yield(m, sample)} t/ha")
    print("Feature importances:", get_feature_importance(m))