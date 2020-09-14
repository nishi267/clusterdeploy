import pickle

from flask import Flask, request

from flasgger import Swagger
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
import os
from sklearn.cluster import KMeans

app = Flask(__name__)
Swagger(app)

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)
picky_in = open("cluster.pkl", "rb")
clusts = pickle.load(picky_in)


@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict_file', methods=["POST"])
def predict_note_file1():
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: The output values

    """
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = classifier.predict(df_test)

    return str(list(prediction))


@app.route('/predict_similar', methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: The output values

    """
    df_test = pd.read_csv(request.files.get("file"), encoding='unicode_escape')
    df_test = df_test.dropna(axis=0, how='any')
    df_test['combine6'] = df_test.iloc[:, 1] + df_test.iloc[:, 2] + df_test.iloc[:, 3]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
    vec.fit(df_test.combine6.values)
    features = vec.transform(df_test.combine6.values)

    clustr = KMeans(init='k-means++', n_clusters=5, n_init=10)
    clustr.fit(features)
    df_test['cluster_labels'] = clustr.labels_
    stry=os.getcwd()
    df_test.to_csv(os.path.join(os.getcwd(), "test_cluster6.csv"))
    return "Check the file is generated" + stry


if __name__ == '__main__':
    app.run()
