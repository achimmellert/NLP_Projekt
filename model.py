from sklearn.preprocessing import StandardScaler
# Training eines Logistic Regression Models mit Sentence-BERT als Vectorizer
# Der Sentence-BERT wird in einer Sklearn-Transformer-Klasse eingebettet, damit eine Pipeline erstellt werden kann

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib


# all-MiniLM-L6-v2 erstellt Vektoren mit 384 Dimensionen
class SentenceBERTVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=256, device="cuda"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y=None):
        self.model_ = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X):
        if not hasattr(self, "model_") or self.model_ is None:
            self.model_ = SentenceTransformer(self.model_name, device=self.device)

        return self.model_.encode(
            X,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True
        )


def get_x_train_trans():
    return pd.read_csv("./data/train_data/X_train.csv", header=None).squeeze("columns")

def get_y_train_trans():
    return pd.read_csv("./data/train_data/y_train.csv", header=None).squeeze("columns")


if __name__ == "__main__":

    pipeline_bert = Pipeline([
        ('sbert', SentenceBERTVectorizer(model_name='all-MiniLM-L6-v2', batch_size=256)),
        ('clf', LogisticRegression(solver="saga",
                                   penalty="l2",
                                   max_iter=40,
                                   random_state=42,
                                   verbose=1
                                   ))
    ])

    X_train = get_x_train_trans()
    y_train = get_y_train_trans()

    pipeline_bert.fit(X_train, y_train)

    joblib.dump(pipeline_bert, "./Models/Streamlit/streamlit_model.joblib")
