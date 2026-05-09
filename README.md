# TextClusterer: Hybrides Graphen-basiertes Clustering

Der `TextClusterer` ist ein spezialisiertes Modul zur automatischen Gruppierung von Textdokumenten. Im Gegensatz zu klassischen Distanz-basierten Verfahren (wie K-Means) nutzt dieses Modul einen **hybriden Graphen-Ansatz**, der sowohl semantische Bedeutung als auch exakte Wortüberschneidungen berücksichtigt.

## Funktionsweise

Der Clustering-Prozess basiert auf einer vierstufigen Pipeline:

1.  **Semantische Ähnlichkeit (FAISS)**: 
    Texte werden mittels `Sentence-Transformers` in Vektoren übersetzt. Ein **HNSW-Index** (Hierarchical Navigable Small World) von FAISS identifiziert effizient die nächsten semantischen Nachbarn im Vektorraum.
2.  **Lexikalische Ähnlichkeit (BM25s)**: 
    Parallel dazu berechnet der BM25-Algorithmus die Ähnlichkeit basierend auf exakten Schlüsselwörtern. Dies stellt sicher, dass Texte mit identischen Fachbegriffen auch dann verknüpft werden, wenn sie semantisch leicht unterschiedlich eingebettet wurden.
3.  **Graphen-Konstruktion**: 
    Es wird ein Netzwerk (Graph) erstellt, in dem Texte die Knoten bilden. Eine Kante zwischen zwei Texten wird gezogen, wenn sie entweder semantisch oder lexikalisch unter den Top-$k$ Nachbarn des jeweils anderen liegen.
4.  **Leiden-Algorithmus**: 
    Anstatt eine feste Cluster-Anzahl vorzugeben, nutzt das Modul den **Leiden-Algorithmus**, um dichte "Communities" innerhalb des Graphen zu finden. Die optimale Anzahl der Cluster wird dabei automatisch durch die Datenstruktur bestimmt.

## Installation

Installiere die benötigten Abhängigkeiten via pip:

```bash
pip install numpy pandas sentence-transformers faiss-cpu bm25s python-igraph leidenalg


--------------------------------------------------

## Die Streamlit App

Die Anwendung dient als Frontend für ein Machine-Learning-Modell, das basierend auf einem Titel und einem Beschreibungstext die wahrscheinlichste Bewertung (Sterne-Rating) vorhersagt.

