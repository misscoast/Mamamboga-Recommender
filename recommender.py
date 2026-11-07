"""
Day 4 - Customer Recommendation Engine (core Python only: csv, collections, pathlib).

Approach: user-based collaborative filtering on purchase *sets*.
  1. Clean the purchase rows (read_purchases).
  2. Compare customers with Jaccard similarity on the sets of products they bought.
  3. Collect products the most similar customers bought that the customer has not.
  4. Rank them with:  score = sum(similarity of supporting neighbours)
                             + POPULARITY_WEIGHT * (share of customers who bought it)
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "purchases.csv"
OUTPUT_PATH = BASE_DIR / "output" / "recommendations.csv"

# How much a product's overall popularity counts relative to neighbour similarity.
# Popularity is a share in [0, 1], so at 0.2 it can break near-ties and lift
# well-liked products, but cannot outweigh a genuinely similar neighbour.
POPULARITY_WEIGHT = 0.2

REQUIRED_COLUMNS = (
    "purchase_id", "customer_id", "customer_name", "product_id",
    "product_name", "category", "purchase_date", "quantity",
)
EXPORT_COLUMNS = (
    "customer_id", "customer_name", "rank", "product_id",
    "product_name", "category", "score", "supporting_customers",
)


class UnknownCustomerError(KeyError):
    """Raised when a customer ID is not in the engine's data."""

    def __str__(self) -> str:  # KeyError would otherwise show repr() with extra quotes
        return str(self.args[0]) if self.args else "unknown customer"


# --------------------------------------------------------------------------- #
# 1. Reading and cleaning
# --------------------------------------------------------------------------- #
def normalise_id(value: str | None) -> str:
    """' c106 ' -> 'C106'. Strips whitespace/hyphens/underscores and upper-cases."""
    return re.sub(r"[\s\-_]+", "", (value or "")).upper()


def read_purchases(path: Path = DATA_PATH) -> tuple[list[dict], list[dict]]:
    """Return (valid_rows, rejected_rows).

    Cleaning rules (each rejected row gets a 'reject_reason'):
      * missing customer_id          -> rejected
      * blank product_id             -> rejected (we cannot tell what was bought,
                                        even if a category is present)
      * quantity not a whole number, or <= 0  -> rejected (nothing was bought)
      * purchase_date not YYYY-MM-DD -> rejected
      * repeated purchase_id         -> rejected; the first occurrence is kept
      * IDs are normalised (strip, upper-case, drop spaces/hyphens), so 'c106'
        is accepted and treated as 'C106' - it is a formatting slip, not bad data.
    """
    valid: list[dict] = []
    rejected: list[dict] = []
    seen_purchase_ids: set[str] = set()

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        for raw in reader:
            row = {k: (v or "").strip() for k, v in raw.items() if k is not None}

            def reject(reason: str) -> None:
                rejected.append({**row, "reject_reason": reason})

            row["customer_id"] = normalise_id(row["customer_id"])
            row["product_id"] = normalise_id(row["product_id"])
            pid = row["purchase_id"]

            if pid and pid in seen_purchase_ids:
                reject("duplicate purchase_id")
                continue
            if not row["customer_id"]:
                reject("missing customer_id")
                continue
            if not row["product_id"]:
                reject("blank product_id")
                continue
            try:
                qty = int(row["quantity"])
            except ValueError:
                reject("invalid quantity")
                continue
            if qty <= 0:
                reject("quantity must be > 0")
                continue
            try:
                date.fromisoformat(row["purchase_date"])
            except ValueError:
                reject("invalid purchase_date")
                continue

            row["quantity"] = qty
            if pid:
                seen_purchase_ids.add(pid)
            valid.append(row)

    return valid, rejected


# --------------------------------------------------------------------------- #
# 2. Similarity
# --------------------------------------------------------------------------- #
def jaccard(a: set, b: set) -> float:
    """|A & B| / |A | B|.  Two empty sets return 0.0 (no evidence of similarity)."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# --------------------------------------------------------------------------- #
# 3. Engine
# --------------------------------------------------------------------------- #
class RecommendationEngine:
    """Recommends products a customer has not bought, based on similar customers."""

    def __init__(self, purchases: list[dict], min_similarity: float = 0.0) -> None:
        self.min_similarity = min_similarity
        self._history: dict[str, set[str]] = {}
        self._customer_names: dict[str, str] = {}
        self._products: dict[str, dict] = {}      # product_id -> {name, category}
        self._buyers: Counter = Counter()         # product_id -> distinct customers

        # One pass over the raw rows; nothing below re-reads them.
        for row in purchases:
            cid, pid = row["customer_id"], row["product_id"]
            self._history.setdefault(cid, set()).add(pid)
            self._customer_names.setdefault(cid, row["customer_name"] or cid)
            self._products.setdefault(
                pid, {"name": row["product_name"], "category": row["category"]}
            )
        for products in self._history.values():
            self._buyers.update(products)

        self._customers = sorted(self._history)

    # -- lookups ------------------------------------------------------------ #
    @property
    def customers(self) -> list[str]:
        return list(self._customers)

    @property
    def products(self) -> list[str]:
        return sorted(self._products)

    def _check(self, customer_id: str) -> None:
        if customer_id not in self._history:
            raise UnknownCustomerError(f"Unknown customer ID: {customer_id!r}")

    def customer_name(self, customer_id: str) -> str:
        self._check(customer_id)
        return self._customer_names[customer_id]

    def product_info(self, product_id: str) -> dict:
        return self._products[product_id]

    def history(self, customer_id: str) -> set[str]:
        self._check(customer_id)
        return set(self._history[customer_id])  # copy: callers cannot mutate our state

    # -- similarity --------------------------------------------------------- #
    def similarity(self, customer_a: str, customer_b: str) -> float:
        self._check(customer_a)
        self._check(customer_b)
        return jaccard(self._history[customer_a], self._history[customer_b])

    def similar_customers(self, customer_id: str, k: int = 3) -> list[tuple[str, float]]:
        """Top-k other customers, best first.

        Excludes the customer, anyone below min_similarity, and anyone with
        similarity 0 (no shared products means not similar, even if
        min_similarity is 0). Ties are broken by customer_id ascending.
        """
        self._check(customer_id)
        scored = []
        for other in self._customers:
            if other == customer_id:
                continue
            sim = self.similarity(customer_id, other)
            if sim > 0 and sim >= self.min_similarity:
                scored.append((other, sim))
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:k]

    # -- candidates and ranking --------------------------------------------- #
    def _popularity_share(self, product_id: str) -> float:
        return self._buyers[product_id] / len(self._customers)

    def candidate_products(self, customer_id: str, k: int = 3) -> dict[str, dict]:
        """Products the k nearest neighbours bought that this customer has not.

        evidence[product_id] = {
            "supporters": [(customer_id, similarity), ...],
            "popularity": distinct buyers overall,
            "popularity_share": popularity / number of customers,
            "fallback": False,
        }

        Fallback: if there are no neighbours (nobody shares a product with this
        customer) or the neighbours have nothing new to offer, return the most
        popular products the customer has not bought, with no supporters and
        fallback=True. A new or unusual customer then still gets sensible,
        clearly-labelled suggestions instead of nothing.
        """
        owned = self.history(customer_id)
        evidence: dict[str, dict] = {}

        for other, sim in self.similar_customers(customer_id, k):
            for pid in self._history[other] - owned:
                entry = evidence.setdefault(pid, {
                    "supporters": [],
                    "popularity": self._buyers[pid],
                    "popularity_share": self._popularity_share(pid),
                    "fallback": False,
                })
                entry["supporters"].append((other, sim))

        if not evidence:
            for pid in self._products:
                if pid not in owned:
                    evidence[pid] = {
                        "supporters": [],
                        "popularity": self._buyers[pid],
                        "popularity_share": self._popularity_share(pid),
                        "fallback": True,
                    }
        return evidence

    def score(self, evidence: dict) -> float:
        """score = sum of supporting neighbours' similarity + POPULARITY_WEIGHT * popularity_share.

        The first term rewards both *more* supporters and *closer* supporters.
        The second adds an independent signal (how widely the product sells).
        """
        similarity_part = sum(sim for _, sim in evidence["supporters"])
        return similarity_part + POPULARITY_WEIGHT * evidence["popularity_share"]

    def recommend(self, customer_id: str, n: int = 3) -> list[dict]:
        """Up to n recommendations, best first. Raises UnknownCustomerError for bad IDs."""
        self._check(customer_id)
        ranked = []
        for pid, ev in self.candidate_products(customer_id).items():
            ranked.append((pid, ev, self.score(ev)))
        # score desc (rounded to hide float noise), more supporters, more popular, product_id
        ranked.sort(key=lambda t: (-round(t[2], 9), -len(t[1]["supporters"]), -t[1]["popularity"], t[0]))

        results = []
        for rank, (pid, ev, sc) in enumerate(ranked[:n], start=1):
            info = self._products[pid]
            results.append({
                "customer_id": customer_id,
                "customer_name": self._customer_names[customer_id],
                "rank": rank,
                "product_id": pid,
                "product_name": info["name"],
                "category": info["category"],
                "score": sc,
                "supporting_customers": [c for c, _ in ev["supporters"]],
                "fallback": ev["fallback"],
            })
        return results

    def recommend_all(self, n: int = 3) -> list[dict]:
        """Rows for every customer. Someone with nothing to recommend (they already
        bought every product) gets one placeholder row with blank product columns."""
        rows: list[dict] = []
        for cid in self._customers:
            recs = self.recommend(cid, n)
            if recs:
                rows.extend(recs)
            else:
                rows.append({
                    "customer_id": cid,
                    "customer_name": self._customer_names[cid],
                    "rank": "", "product_id": "", "product_name": "",
                    "category": "", "score": "", "supporting_customers": [],
                    "fallback": False,
                })
        return rows

    def export_csv(self, rows: list[dict], path: Path = OUTPUT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    "customer_id": r["customer_id"],
                    "customer_name": r["customer_name"],
                    "rank": r["rank"],
                    "product_id": r["product_id"],
                    "product_name": r["product_name"],
                    "category": r["category"],
                    "score": f"{r['score']:.3f}" if r["score"] != "" else "",
                    "supporting_customers": ";".join(r["supporting_customers"]),
                })
