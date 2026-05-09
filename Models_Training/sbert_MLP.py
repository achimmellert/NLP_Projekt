import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import time
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Union, List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau


DATA_DIR = Path("./data/train_data")
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
SEED = 42


class MLPClassifier(nn.Module):
    """
    MLP mit Dropout und Layer Normalization.
    """

    def __init__(self, input_dim=EMBEDDING_DIM, hidden_dim=256, num_classes=5, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes)
        )


    def forward(self, x):
        return self.net(x.float())


class SBERTVectorizer(BaseEstimator, TransformerMixin):
    """
    Wrapper für SBERT mit Lazy Loading.
    """

    def __init__(self, model_name: str = SBERT_MODEL_NAME, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_ = None


    def _get_model(self):
        if self.model_ is None:
            self.model_ = SentenceTransformer(self.model_name, device=self.device)
        return self.model_


    def fit(self, X, y=None):
        self._get_model()
        return self


    def transform(self, X):
        X_list = X.tolist() if isinstance(X, pd.Series) else X
        return self._get_model().encode(
            X_list,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True
        )


    def __getstate__(self):
        state = self.__dict__.copy()
        state['model_'] = None
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)


def train_pytorch_pipeline():

    X_train = pd.read_csv(DATA_DIR / "X_train.csv", header=None).squeeze("columns")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv", header=None).squeeze("columns")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train).astype(np.int64)
    num_classes = len(le.classes_)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = NeuralNetClassifier(
        module=MLPClassifier,
        module__input_dim=EMBEDDING_DIM,
        module__num_classes=num_classes,
        module__dropout=0.3,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=0.001,
        max_epochs=100,
        batch_size=128,
        device=device,
        iterator_train__shuffle=True,
        callbacks=[
            ('early_stopping', EarlyStopping(patience=10, monitor='valid_loss')),
            ('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss')),
            ('progress_bar', ProgressBar()),
        ],
    )

    pipeline = Pipeline([
        ("embedder", SBERTVectorizer()),
        ("classifier", net)
    ])

    mlflow.set_experiment("PyTorch_MLP_Classification")

    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="MLP_SBERT_Optimized") as run:
        print(f"Training gestartet auf {device}...")

        start_time = time.time()
        pipeline.fit(X_train, y_encoded)
        duration_min = (time.time() - start_time) / 60

        mlflow.log_metric("total_duration_minutes", duration_min)

        label_mapping = {int(i): label for i, label in enumerate(le.classes_)}
        with open("label_mapping.json", "w") as f:
            json.dump(label_mapping, f)
        mlflow.log_artifact("label_mapping.json", artifact_path="metadata")

        print(f"Training beendet in {duration_min:.2f} Minuten.")

        return run.info.run_id


if __name__ == "__main__":
    train_pytorch_pipeline()
