import pandas as pd
import torch
import json
import time
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Union, List
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("./data/train_data")
MODEL_NAME = "all-MiniLM-L6-v2"
SEED = 42


class SBERTVectorizer(BaseEstimator, TransformerMixin):
    """
    Wrapper für Sentence-Transformer
    """

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 64, device: str = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None

    def _get_model(self):
        if self.model_ is None:
            self.model_ = SentenceTransformer(self.model_name, device=self.device)
        return self.model_

    def fit(self, X, y=None):
        self._get_model()
        return self

    def transform(self, X: Union[pd.Series, List[str]]):
        X_list = X.tolist() if isinstance(X, pd.Series) else X
        model = self._get_model()

        return model.encode(
            X_list,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def __getstate__(self):
        """Verhindert, dass das schwere Modell-Objekt mitgepickelt wird."""
        state = self.__dict__.copy()
        state['model_'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def load_data():
    X = pd.read_csv(DATA_DIR / "X_train.csv", header=None).squeeze("columns")
    y = pd.read_csv(DATA_DIR / "y_train.csv", header=None).squeeze("columns")
    return X, y


def train_bert_pipeline():
    X_train, y_train = load_data()

    pipeline = Pipeline([
        ("sbert", SBERTVectorizer()),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="l2",
            max_iter=500,
            random_state=SEED,
            n_jobs=-1,
            verbose=0
        ))
    ])

    mlflow.set_experiment("Text_Classification_SBERT")

    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="SBERT_LogReg_Modern") as run:
        print(f"Training läuft auf: {pipeline.named_steps['sbert'].device}")

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        duration_min = (time.time() - start_time) / 60

        mlflow.log_metric("duration_minutes", duration_min)

        label_path = Path("class_labels.json")
        with open(label_path, "w") as f:
            json.dump(pipeline.classes_.tolist(), f)
        mlflow.log_artifact(str(label_path), artifact_path="metadata")

        print(f"Fertig! Training dauerte {duration_min:.2f} Minuten.")
        return run.info.run_id


if __name__ == "__main__":
    train_bert_pipeline()
