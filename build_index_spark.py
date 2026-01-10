# =========================
# build_index_spark.py
# IR Project – Minimum Requirements (FINAL, CLEAN)
# =========================

import re
import hashlib
from collections import defaultdict

from pyspark.sql import SparkSession
from inverted_index_gcp import InvertedIndex

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BUCKET_NAME = "roi-ir-bucket-1919"
BASE_DIR = "."
INDEX_NAME = "index"
NUM_BUCKETS = 124

WIKI_PARQUET = "gs://roi-ir-bucket-1919/multistream10_preprocessed.parquet"

# --------------------------------------------------
# Tokenization
# --------------------------------------------------

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

ENGLISH_STOPWORDS = frozenset([
    "the","and","of","to","in","is","it","that","as","for","with","was","on",
    "by","at","from","this","be","or","an","are","which","but","not","have",
    "has","had","were","their","they","its","if","about","into","than"
])

CORPUS_STOPWORDS = {"category","references","also","links","external","see","thumb"}
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
# MapReduce helpers
# --------------------------------------------------

def word_count(text, doc_id):
    tf = defaultdict(int)
    for token in tokenize(text):
        tf[token] += 1
    return [(token, (doc_id, freq)) for token, freq in tf.items()]

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    spark = (
        SparkSession.builder
        .appName("IR-Build-Index")
        .getOrCreate()
    )

    # Load Wikipedia parquet
    df = spark.read.parquet(WIKI_PARQUET)
    doc_text_pairs = df.select("text", "id").rdd

    # TF
    word_counts = doc_text_pairs.flatMap(
        lambda x: word_count(x[0], x[1])
    )

    # postings: term -> [(doc_id, tf), ...]
    postings = word_counts.groupByKey().mapValues(list)

    # filter rare terms (DF > 50)
    postings_filtered = postings.filter(lambda x: len(x[1]) > 50)

    # DF dictionary
    w2df = postings_filtered.mapValues(len)
    w2df_dict = w2df.collectAsMap()

    # bucket by hash
    token_bucket = postings_filtered.map(
        lambda x: (token2bucket_id(x[0]), (x[0], x[1]))
    )

    bucketed = token_bucket.groupByKey()

    # 🔑 WRITE POSTINGS (NO COLLECT, NO RETURN)
    bucketed.foreach(
        lambda bucket: InvertedIndex.write_a_posting_list(
            (bucket[0], list(bucket[1])),
            BASE_DIR,
            BUCKET_NAME
        )
    )

    # Write index globals
    inverted = InvertedIndex()
    inverted.df = dict(w2df_dict)
    inverted.N = df.count()

    inverted.write_index(BASE_DIR, INDEX_NAME, BUCKET_NAME)

    spark.stop()
    print("✅ Full index written successfully to GCS")

# --------------------------------------------------

if __name__ == "__main__":
    main()
