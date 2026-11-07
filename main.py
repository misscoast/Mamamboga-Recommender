"""Runs the recommendation engine end to end:  python main.py"""
from __future__ import annotations

from collections import Counter

from recommender import (
    DATA_PATH, OUTPUT_PATH, RecommendationEngine, read_purchases,
)

N_RECS = 3


def print_similarity_matrix(engine: RecommendationEngine, count: int = 5) -> None:
    ids = engine.customers[:count]
    print("Similarity matrix (Jaccard), first five customers")
    print("        " + "".join(f"{c:>7}" for c in ids))
    for a in ids:
        print(f"{a:>7} " + "".join(f"{engine.similarity(a, b):7.2f}" for b in ids))
    print()


def print_customer(engine: RecommendationEngine, customer_id: str) -> None:
    bought = sorted(engine.history(customer_id))
    print(f"Customer {customer_id} ({engine.customer_name(customer_id)}) bought: {{{', '.join(bought)}}}")
    print("Most similar customers:")
    neighbours = engine.similar_customers(customer_id, k=3)
    if not neighbours:
        print("  (none - nobody shares a product with this customer)")
    for cid, sim in neighbours:
        print(f"  {cid}  jaccard={sim:.2f}")
    print("Recommendations:")
    recs = engine.recommend(customer_id, N_RECS)
    if not recs:
        print("  (nothing to recommend - already bought every product)")
    for r in recs:
        if r["fallback"]:
            why = "it is popular overall (no similar customers had anything new)"
        else:
            n_sup = len(r["supporting_customers"])
            why = f"{n_sup} similar customer{'s' if n_sup != 1 else ''} bought it"
        print(f"  {r['rank']}. {r['product_name']} ({r['category']})  score={r['score']:.3f}  because {why}")
    print()


def main() -> None:
    valid, rejected = read_purchases(DATA_PATH)
    engine = RecommendationEngine(valid)

    print_similarity_matrix(engine)
    for cid in ("C101", "C113"):
        print_customer(engine, cid)

    rows = engine.recommend_all(n=N_RECS)
    engine.export_csv(rows, OUTPUT_PATH)

    written = sum(1 for r in rows if r["product_id"])
    print("Rejected rows:", dict(Counter(r["reject_reason"] for r in rejected)))
    print()
    print("## Run summary")
    print(f"Purchases read: {len(valid) + len(rejected)} | used: {len(valid)} | rejected: {len(rejected)}")
    print(f"Customers: {len(engine.customers)} | Products: {len(engine.products)} | Recommendations written: {written}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
