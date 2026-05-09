import pandas as pd
import spacy
import re
import json
import time
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import List, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


DATA_DIR = Path("./data/train_data")
MODEL_NAME = "en_core_web_sm"
SEED = 42


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Text-Cleaner unter Verwendung von spaCy Multiprocessing.
    """

    def __init__(self, model_name: str = MODEL_NAME, n_process: int = -1):
        self.model_name = model_name
        self.n_process = n_process

        try:

            self.nlp = spacy.load(self.model_name, disable=["parser", "ner", "lemmatizer"])
            self.nlp.add_pipe("lemmatizer", config={"mode": "rule"})

        except OSError:
            raise OSError(
                f"Modell {self.model_name} nicht gefunden. Bitte mit 'python -m spacy download {self.model_name}' installieren.")


    def fit(self, X, y=None):
        return self


    def _pre_clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower().strip()
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)

        return re.sub(r"\s+", " ", text)


    def transform(self, X: Union[pd.Series, List[str]]) -> pd.Series:
        X_series = pd.Series(X) if not isinstance(X, pd.Series) else X

        # Vorreinigung (Regex)
        cleaned_input = X_series.apply(self._pre_clean)

        # spaCy Batch Processing
        processed_docs = []
        for doc in self.nlp.pipe(cleaned_input, n_process=self.n_process, batch_size=500):
            tokens = [
                token.lemma_ for token in doc
                if not token.is_stop and token.is_alpha and len(token.lemma_) > 1
            ]
            processed_docs.append(" ".join(tokens))

        return pd.Series(processed_docs, index=X_series.index)


def load_data():
    X = pd.read_csv(DATA_DIR / "X_train.csv", header=None).squeeze("columns")
    y = pd.read_csv(DATA_DIR / "y_train.csv", header=None).squeeze("columns")
    return X, y

###########################
# MAIN TRAINING PIPELINE
###########################

def train():

    X, y = load_data()

    pipeline = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=0.001,
            max_df=0.95,
            sublinear_tf=True  # Best Practice für LogReg
        )),
        ("lr", LogisticRegression(
            solver="saga",
            penalty="l2",
            max_iter=1000,
            random_state=SEED,
            n_jobs=-1
        ))
    ])

    # MLflow Tracking
    mlflow.set_experiment("Text_Classification_Logistic_Regression")

    # Automatisiertes Logging (loggt Params, Metrics und Model-Signature)
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="TFIDF_LogReg_Optimized") as run:
        print(f"Starte Training für Run: {run.info.run_id}")

        start_time = time.time()
        pipeline.fit(X, y)
        duration_min = (time.time() - start_time) / 60

        mlflow.log_metric("manual_duration_minutes", duration_min)

        class_labels = {"classes": pipeline.classes_.tolist()}
        label_path = Path("class_labels.json")
        with open(label_path, "w") as f:
            json.dump(class_labels, f)

        mlflow.log_artifact(str(label_path), artifact_path="metadata")

        print(f"Training abgeschlossen in {duration_min:.2f} Minuten.")


if __name__ == "__main__":
    train()
