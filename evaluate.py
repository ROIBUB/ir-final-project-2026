import json
import requests
import random

BASE_URL = "http://127.0.0.1:8080/search"
RANDOM_SEED = 42
TRAIN_RATIO = 0.7


# =========================
# Metrics – EXACTLY as course
# =========================

def precision_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(predicted_list) == 0:
        return 0.0
    return round(len([d for d in predicted_list if d in true_set]) / len(predicted_list), 3)


def recall_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(true_set) < 1:
        return 1.0
    return round(len([d for d in predicted_list if d in true_set]) / len(true_set), 3)


def f1_at_k(true_list, predicted_list, k):
    p = precision_at_k(true_list, predicted_list, k)
    r = recall_at_k(true_list, predicted_list, k)
    if p == 0.0 or r == 0.0:
        return 0.0
    return round(2.0 / (1.0 / p + 1.0 / r), 3)


def results_quality(true_list, predicted_list):
    p5 = precision_at_k(true_list, predicted_list, 5)
    f1_30 = f1_at_k(true_list, predicted_list, 30)
    if p5 == 0.0 or f1_30 == 0.0:
        return 0.0
    return round(2.0 / (1.0 / p5 + 1.0 / f1_30), 3)

def debug_single_query(query, true_docs):
    print("\n" + "=" * 80)
    print("DEBUG SINGLE QUERY")
    print("QUERY:", query)
    print("TRUE DOCS (sample):", true_docs[:10])

    response = requests.get(BASE_URL, params={"query": query})
    results = response.json()
    predicted_docs = [int(doc_id) for doc_id, _ in results]

    print("PREDICTED DOCS (top 10):", predicted_docs[:10])

    overlap = set(true_docs) & set(predicted_docs[:40])
    print("OVERLAP in top 40:", len(overlap))
    print("OVERLAP DOC IDS:", list(overlap)[:10])

    print("P@5:", precision_at_k(true_docs, predicted_docs, 5))
    print("F1@30:", f1_at_k(true_docs, predicted_docs, 30))
    print("RESULTS_QUALITY:", results_quality(true_docs, predicted_docs))
    print("=" * 80)


# =========================
# Evaluation
# =========================

def run_eval(queries_subset, label):
    scores = []

    print(f"\n==============================")
    print(f"RUNNING EVAL: {label}")
    print(f"==============================")

    for query, true_docs in queries_subset.items():
        print("\n------------------------------")
        print("QUERY:", query)
        print("TRUE DOCS COUNT:", len(true_docs))
        print("TRUE DOCS SAMPLE:", true_docs[:10])

        response = requests.get(BASE_URL, params={"query": query})
        results = response.json()

        predicted_docs = [int(doc_id) for doc_id, _ in results]
        true_docs = [int(d) for d in true_docs]

        print("RAW RESULTS COUNT:", len(predicted_docs))
        print("TOP 10 PREDICTED:", predicted_docs[:10])

        p5 = precision_at_k(true_docs, predicted_docs, 5)
        f1_30 = f1_at_k(true_docs, predicted_docs, 30)
        rq = results_quality(true_docs, predicted_docs)

        overlap = set(true_docs) & set(predicted_docs[:40])
        print("OVERLAP (top 40):", len(overlap))

        print("P@5:", p5)
        print("F1@30:", f1_30)
        print("RESULTS_QUALITY:", rq)

        scores.append(rq)

    avg_score = round(sum(scores) / len(scores), 3)
    print(f"\n===== {label} RESULTS =====")
    print("Queries:", len(scores))
    print("Average results_quality:", avg_score)
    return avg_score


def evaluate_with_split(queries_file):
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    items = list(queries.items())
    random.seed(RANDOM_SEED)
    random.shuffle(items)

    split_idx = int(len(items) * TRAIN_RATIO)
    train_items = dict(items[:split_idx])
    val_items = dict(items[split_idx:])

    # DEBUG: run single query first
    first_query, first_true_docs = items[0]
    debug_single_query(first_query, [int(x) for x in first_true_docs])

    # Full evaluation
    run_eval(train_items, "TRAIN")
    run_eval(val_items, "VALIDATION")


if __name__ == "__main__":
    evaluate_with_split("queries_train.json")