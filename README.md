# Information Retrieval Final Project

---

## Authors
Roi Bublil & Hadar Shir 

## Project Overview
This project implements an Information Retrieval search engine over the English Wikipedia corpus. The project was made in "IR" course in the department of "Software and Information Systems Engineering" in "Ben-Gurion University"  
This engine was built according to our theoretical inforamtion we learned thrughout this course, and it's built according to entire english Wikipedia as a corpus, thats corpus size of 6,348,910 documents.

---

## System Description
The inverted index is constructed offline using Apache Spark and stored in Google Cloud Storage.  
At query time, the system loads the index and retrieves posting lists directly from cloud storage.

The search functionality is exposed via a single REST endpoint (`/search`), which accepts free-text queries and returns up to 100 ranked Wikipedia documents.  
Queries are tokenized using a custom tokenizer with stopword removal, and only terms that appear in the index are considered.

Document ranking is based on a combination of term-based scoring methods and global importance signals.  
The system computes TF-IDF scores with cosine-style normalization and applies a BM25-style component to model term frequency saturation.  
In addition, PageRank values computed over the Wikipedia link graph are used as a secondary ranking signal.

A single fixed retrieval configuration is used for all queries, without query-time caching or external APIs.

---

## Repository Structure
- `inverted_index_gcp.py` – Inverted index implementation with support for Google Cloud Storage  
- `search_frontend.py` – Flask-based search server  
- `run_frontend_in_gcp.sh` – Script for running the server on GCP  
- `queries_train.json` – Query set used for evaluation
-  `index_building.ipynb` – Jupyter Notebook for Building an Index  
 

---

## Index Files and Storage
Due to their size, the index files are not stored in the Git repository. All index artifacts are stored in the following Google Cloud Storage bucket:
https://storage.googleapis.com/roi-ir-bucket-1919

---

## How to Run the Search Server
1. Ensure that the index files are available in the Google Cloud Storage bucket.
2. Run the search server:
   ```bash
   python search_frontend.py
3. The server will be available at: http://34.30.48.116:8080
Example Query : curl "http://34.30.48.116:8080/search?query=dana+international+eurovision"

---

## Evaluation
System evaluation was performed using the provided query set and the evaluation metrics defined in the course materials, including Precision@5, F1@30, and results_quality.

