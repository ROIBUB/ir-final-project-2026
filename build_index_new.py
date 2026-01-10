# =========================
# build_index_small_gcs.py
# IR Project – Small Index Builder (GCS)
# =========================

import re
import hashlib
from collections import defaultdict

from inverted_index_gcp import InvertedIndex

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BUCKET_NAME = "roi-ir-bucket-1919" # ← לשנות
BASE_DIR = "."
INDEX_NAME = "index_small"         # שם האינדקס בבאקט
NUM_BUCKETS = 124

# --------------------------------------------------
# Tokenization (זהה לבנייה ולחיפוש)
# --------------------------------------------------

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

ENGLISH_STOPWORDS = frozenset([
    "the","and","of","to","in","is","it","that","as","for","with","was","on",
    "by","at","from","this","be","or","an","are","which","but","not","have",
    "has","had","were","their","they","its","if","about","into","than"
])

CORPUS_STOPWORDS = {
    "category","references","also","links","external","see","thumb"
}

ALL_STOPWORDS = ENGLISH_STOPWORDS.union(CORPUS_STOPWORDS)

def tokenize(text):
    tokens = [m.group() for m in RE_WORD.finditer(text.lower())]
    return [t for t in tokens if t not in ALL_STOPWORDS]

# --------------------------------------------------
# Hashing & Bucketing
# --------------------------------------------------

def _hash(token):
    return hashlib.blake2b(token.encode("utf8"), digest_size=5).hexdigest()

def token2bucket_id(token):
    return int(_hash(token), 16) % NUM_BUCKETS

# --------------------------------------------------
# Index construction
# --------------------------------------------------

def build_index(documents):
    """
    documents: iterable of (doc_id, text)
    """

    posting_lists = defaultdict(list)
    df = defaultdict(int)

    # TF per document
    for doc_id, text in documents:
        tf = defaultdict(int)
        for token in tokenize(text):
            tf[token] += 1

        for token, freq in tf.items():
            posting_lists[token].append((doc_id, freq))

    # Sort postings & compute DF
    for token, pl in posting_lists.items():
        pl.sort(key=lambda x: x[0])
        df[token] = len(pl)

    # # Write posting lists to GCS
    # posting_locs = defaultdict(list)
    # for token, pl in posting_lists.items():
    #     bucket_id = token2bucket_id(token)
    #     locs = InvertedIndex.write_a_posting_list(
    #         (bucket_id, [(token, pl)]),
    #         BASE_DIR,
    #         BUCKET_NAME
    #     )
    #     posting_locs[token].extend(locs[token])
    for token, pl in posting_lists.items():
        bucket_id = token2bucket_id(token)
        InvertedIndex.write_a_posting_list(
            (bucket_id, [(token, pl)]),
            BASE_DIR,
            BUCKET_NAME
        )

    # Build inverted index object
    # inverted = InvertedIndex()
    # inverted.df = dict(df)
    # inverted.posting_locs = dict(posting_locs)
    # inverted.N = len(documents)

    inverted = InvertedIndex()
    inverted.df = dict(df)
    inverted.N = len(documents)
    inverted.write_index(BASE_DIR, INDEX_NAME, BUCKET_NAME)

    # 🚨 כאן החיבור לבאקט (כמו עבודה 3)
    inverted.write_index(BASE_DIR, INDEX_NAME, BUCKET_NAME)


    return inverted

# --------------------------------------------------
# Example run (small corpus)
# --------------------------------------------------

if __name__ == "__main__":

    docs = [
        (1, "information retrieval is fun"),
        (2, "retrieval of information"),
        (3, "this course is about information retrieval"),
    ]

    build_index(docs)
    print("Small index written to GCS successfully.")
