from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

def train_model(X_train, y_train, preporcessor):
    model = Pipeline(steps=[
        ("preprocessor", preporcessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    
    model.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    
    model_path = os.path.join("models", "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model Saved At: {model_path}")
    return model