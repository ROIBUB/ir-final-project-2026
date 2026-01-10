from flask import Flask, request, jsonify
import os
import re
from google.cloud import storage
from collections import defaultdict
import numpy as np
import nltk
from nltk.corpus import stopwords
from inverted_index_gcp import InvertedIndex

import pickle

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# --- Block 2: Text Processing and Tokenization (Updated from Assignment 2) ---

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
ALL_STOPWORDS = english_stopwords.union(corpus_stopwords)

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

with open("index.pkl", "rb") as f:
    index = pickle.load(f)

with open("doc2title_dict.pkl", "rb") as f:
    doc2title = pickle.load(f)

with open("pagerank.pkl", "rb") as f:
    pagerank = pickle.load(f)

with open("avg_doc_len.pkl", "rb") as f:
    AVG_DL = pickle.load(f)


def tokenize(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [token for token in tokens if token not in ALL_STOPWORDS]


# --- Block 3: Loading Indexes and Global Data ---

BUCKET_NAME = os.environ.get("BUCKET_NAME", "roi-ir-bucket-1919")


def load_index(base_dir, name, bucket_name):
    return InvertedIndex.read_index(base_dir, name, bucket_name)

def load_pickle(path, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    with blob.open("rb") as f:
        return pickle.load(f)



def load_pickle(path, bucket_name):
    if bucket_name == 'roi-ir-bucket-1919' or bucket_name is None:
        with open(path, "rb") as f:
            return pickle.load(f)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    with blob.open("rb") as f:
        return pickle.load(f)



INDEX_BODY = None
BUCKET_NAME = "roi-ir-bucket-1919"
# doc2title = load_pickle("doc2title_dict.pkl", BUCKET_NAME)
# pagerank = load_pickle("pagerank.pkl", BUCKET_NAME)
# AVG_DL = load_pickle("avg_doc_len.pkl", BUCKET_NAME)
#BASE_DIR = "postings_gcp/"
BASE_DIR = "postings_gcp"
INDEX_BODY = InvertedIndex.read_index(
    base_dir= BASE_DIR,
    name="index",
    bucket_name=BUCKET_NAME
)


# --- Block 4: Helper Functions for Retrieval Logic ---

def get_top_n(sim_dict, n=100):
    return sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)[:n]


def get_posting_iter(index, words, bucket_name, directory):
    #  (df = document frequency)
    valid_words = [w for w in words if w in index.df]

    current_bucket = bucket_name

    for w in valid_words:
        yield w, index.read_a_posting_list(directory, w, current_bucket)


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    include_title = request.args.get('include_title', 'false') == 'true'  # If True, search in title as well

    if not query:
        return jsonify(res)

    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    scores_tfidf = defaultdict(float)
    scores_bm25 = defaultdict(float)
    doc_norms = defaultdict(float)

    N = len(doc2title)
    k1 = 1.5
    b = 0.75


    # -------- Iterate over postings --------
    for term, posting_list in get_posting_iter(
            INDEX_BODY, tokens, BUCKET_NAME, BASE_DIR):

        df = INDEX_BODY.df[term]
        idf_tfidf = np.log10(N / df)
        idf_bm25 = np.log10((N - df + 0.5) / (df + 0.5))

        for doc_id, tf in posting_list:
            # ---- TF-IDF ----
            w_td = tf * idf_tfidf
            scores_tfidf[doc_id] += w_td * idf_tfidf
            doc_norms[doc_id] += w_td ** 2

            # ---- BM25 ----
            dl = AVG_DL
            B = 1 - b + b * 1.0

            bm25_score = idf_bm25 * ((k1 + 1) * tf) / (k1 * B + tf)
            scores_bm25[doc_id] += bm25_score

    # -------- Cosine normalization --------
    for doc_id in scores_tfidf:
        scores_tfidf[doc_id] /= np.sqrt(doc_norms[doc_id])

    # -------- PageRank boost (optional) --------
    for doc_id in scores_tfidf:
        pr = pagerank.get(doc_id, 0.0)
        scores_tfidf[doc_id] *= (1 + 0.1 * np.log10(pr + 1))

    # -------- Combine scores --------
    final_scores = {}
    for doc_id in scores_tfidf:
        final_scores[doc_id] = (
                0.5 * scores_tfidf.get(doc_id, 0) +
                0.5 * scores_bm25.get(doc_id, 0)
        )

    # -------- Top 100 --------
    top_n = get_top_n(final_scores, n=100)

    res = [
        (doc_id, doc2title.get(doc_id, ""))
        for doc_id, _ in top_n
    ]
    return jsonify(res)

def run(**options):
    app.run(**options)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    print("NUMBER OF TERMS IN INDEX:", len(INDEX_BODY.df))
    print("SAMPLE TERMS:", list(INDEX_BODY.df.keys())[:20])
    app.run(host='0.0.0.0', port=8080, debug=True)


