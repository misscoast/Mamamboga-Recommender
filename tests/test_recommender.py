import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommender import (  # noqa: E402
    RecommendationEngine, UnknownCustomerError, jaccard, normalise_id, read_purchases,
)

HEADER = "purchase_id,customer_id,customer_name,product_id,product_name,category,purchase_date,quantity\n"


def row(cid, pid, n=1, qty=1, name="X"):
    return {"customer_id": cid, "customer_name": name, "product_id": pid,
            "product_name": f"Prod {pid}", "category": "Cat", "quantity": qty}


def make_engine(**kw):
    data = [
        row("C1", "P1"), row("C1", "P2"),
        row("C2", "P1"), row("C2", "P2"), row("C2", "P3"),
        row("C3", "P2"), row("C3", "P4"),
        row("C4", "P9"),  # shares nothing with anyone
    ]
    return RecommendationEngine(data, **kw)


class JaccardTests(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(jaccard({1, 2}, {2, 3}), 1 / 3)

    def test_identical_is_one(self):
        self.assertEqual(jaccard({1, 2}, {1, 2}), 1.0)

    def test_both_empty_is_zero(self):
        self.assertEqual(jaccard(set(), set()), 0.0)

    def test_one_empty_is_zero(self):
        self.assertEqual(jaccard({1}, set()), 0.0)


class ReadTests(unittest.TestCase):
    def read(self, body):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.csv"
            p.write_text(HEADER + body, encoding="utf-8")
            return read_purchases(p)

    def test_rules(self):
        valid, rejected = self.read(
            "T1,C1,A,P1,Milk,Dairy,2026-10-01,2\n"
            "T1,C1,A,P1,Milk,Dairy,2026-10-01,2\n"     # repeated
            "T2,C1,A,,,Fresh,2026-10-01,1\n"            # blank product
            "T3,C1,A,P2,Eggs,Fresh,2026-10-01,0\n"      # zero qty
            "T4,c2,B,P1,Milk,Dairy,2026-10-01,1\n"      # lower-case id
            "T5,C2,B,P3,Tea,Bev,not-a-date,1\n"         # bad date
            "T6,C2,B,P3,Tea,Bev,2026-10-01,abc\n"       # bad qty
        )
        self.assertEqual([r["purchase_id"] for r in valid], ["T1", "T4"])
        self.assertEqual(valid[1]["customer_id"], "C2")
        self.assertEqual(len(rejected), 5)

    def test_normalise_id(self):
        self.assertEqual(normalise_id(" c-106 "), "C106")


class EngineTests(unittest.TestCase):
    def test_similarity_symmetric_and_diagonal(self):
        e = make_engine()
        self.assertEqual(e.similarity("C1", "C1"), 1.0)
        self.assertEqual(e.similarity("C1", "C2"), e.similarity("C2", "C1"))

    def test_similar_customers_excludes_self_and_zero(self):
        e = make_engine()
        ids = [c for c, _ in e.similar_customers("C1", k=10)]
        self.assertNotIn("C1", ids)
        self.assertNotIn("C4", ids)

    def test_min_similarity_filters(self):
        e = make_engine(min_similarity=0.5)
        self.assertEqual([c for c, _ in e.similar_customers("C1")], ["C2"])

    def test_tie_break_is_by_customer_id(self):
        e = RecommendationEngine([row("A", "P1"), row("B", "P1"), row("C", "P1")])
        self.assertEqual([c for c, _ in e.similar_customers("A")], ["B", "C"])

    def test_candidates_exclude_owned(self):
        e = make_engine()
        cands = e.candidate_products("C1")
        self.assertNotIn("P1", cands)
        self.assertNotIn("P2", cands)
        self.assertEqual(set(cands), {"P3", "P4"})

    def test_fallback_for_customer_without_neighbours(self):
        e = make_engine()
        recs = e.recommend("C4", n=2)
        self.assertEqual(len(recs), 2)
        self.assertTrue(all(r["fallback"] for r in recs))
        self.assertEqual(recs[0]["product_id"], "P2")  # most bought product

    def test_unknown_customer_raises(self):
        e = make_engine()
        for call in (e.history, e.recommend, e.similar_customers):
            with self.assertRaises(KeyError):
                call("NOPE")
        with self.assertRaises(UnknownCustomerError):
            e.similarity("C1", "NOPE")

    def test_recommend_is_deterministic_and_ranked(self):
        e = make_engine()
        a, b = e.recommend("C1", 3), e.recommend("C1", 3)
        self.assertEqual(a, b)
        scores = [r["score"] for r in a]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_history_returns_copy(self):
        e = make_engine()
        e.history("C1").add("ZZZ")
        self.assertNotIn("ZZZ", e.history("C1"))

    def test_export_has_every_customer(self):
        e = make_engine()
        rows = e.recommend_all(3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out" / "r.csv"
            e.export_csv(rows, p)
            with open(p, newline="", encoding="utf-8") as fh:
                 ids = {r["customer_id"] for r in csv.DictReader(fh)}
        self.assertEqual(ids, set(e.customers))

    def test_customer_with_everything_gets_blank_row(self):
        e = RecommendationEngine([row("A", "P1"), row("B", "P1")])
        rows = e.recommend_all(3)
        self.assertEqual([r["product_id"] for r in rows], ["", ""])


if __name__ == "__main__":
    unittest.main()
