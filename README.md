# Day 4 - Customer Recommendation Engine

A customer-to-customer recommender for grocery purchases, written in core Python
(`csv`, `collections`, `pathlib`, `re`, `datetime` only - no ML libraries).

## Approach

1. **Clean** the purchase rows (`read_purchases`).
2. **Build once** (`RecommendationEngine.__init__`): `customer -> set of products`,
   `product -> name/category`, and `product -> number of distinct buyers`.
   No later method touches the raw rows.
3. **Compare** customers with Jaccard similarity on their product sets.
4. **Pick neighbours**: the top-3 most similar other customers.
5. **Collect candidates**: products the neighbours bought that the customer has not,
   remembering which neighbours support each product.
6. **Rank** with the scoring rule below and keep the top 3.

## How to run

```bash
python main.py                       # prints results and writes output/recommendations.csv
python -m unittest discover -s tests # 17 tests
```

Python 3.9+; no packages to install. Library use:

```python
from recommender import RecommendationEngine, read_purchases
valid, rejected = read_purchases()
RecommendationEngine(valid).recommend_all(n=3)
```

## Data-cleaning rules

| Situation | Rule |
|---|---|
| Repeated row (`T9010` appears twice) | Keep the first, reject later rows with the same `purchase_id` |
| Blank `product_id` (`T9020`) | Rejected - we can't tell what was bought, even though a category is present |
| Quantity 0 (`T9033`) | Rejected - nothing was bought. Non-numeric or negative quantities are rejected too |
| Inconsistent ID (`c106`) | **Kept**, normalised to `C106` (strip, upper-case, drop spaces/hyphens). It is a formatting slip, not bad data |
| Missing `customer_id`, or a date that is not `YYYY-MM-DD` | Rejected |

Result on the supplied file: 63 rows read, 60 used, 3 rejected.

## Similarity and neighbours

* `jaccard(a, b) = |A & B| / |A | B|`; two empty sets give `0.0` (no evidence of similarity).
* Neighbours exclude the customer themself, anyone below `min_similarity`, and anyone
  with similarity `0` (sharing nothing is not "similar").
* Ties are broken by customer ID ascending, so results are deterministic.

## Scoring rule

```
score = sum(similarity of each neighbour who bought the product)
        + 0.2 * (customers who bought it / all customers)
```

* The first term combines **similarity** with the **number of supporters**: many close
  neighbours beat one.
* The second term is **popularity**. It is capped at 0.2, so it breaks near-ties and
  nudges well-liked products up, but cannot beat a genuinely similar neighbour.
* Final ties: more supporters, then more popular, then product ID.

### Worked example - Wanjiru (C101), Festive Bread (P03)

Wanjiru bought `{P04, P08}`. Her three nearest neighbours:

| Neighbour | Jaccard | Bought P03? |
|---|---|---|
| C117 | 2/4 = 0.500 | yes |
| C109 | 1/3 = 0.333 | yes |
| C113 | 1/6 = 0.167 | yes |

* Similarity part: 0.500 + 0.333 + 0.167 = **1.000**
* Popularity part: 7 of 20 customers bought P03, so 0.2 x 0.35 = **0.070**
* **Score = 1.070**, which puts it first. Pampers (supported only by C117, 0.500 + 0.2 x 0.30 = 0.560) is second.

## Fallback for customers without useful neighbours

If a customer has no neighbours, or their neighbours have nothing new for them, the
engine recommends the **most popular products they have not bought** (popularity-only
score, `supporting_customers` left blank). This gives a sensible "bestsellers you
haven't tried" answer rather than nothing. In the supplied data this happens for
customers such as C107 and C108, who already own most of what their neighbours buy.

## Other behaviour

* Unknown customer IDs raise `UnknownCustomerError` (a `KeyError` subclass) with a clear message.
* A customer who already bought every product gets one row with blank product columns,
  so nobody disappears from the CSV. `recommend` returns *up to* `n` items: a customer
  may get fewer than 3 if few candidates exist.
* `output/recommendations.csv` columns: `customer_id, customer_name, rank, product_id,
  product_name, category, score (3 dp), supporting_customers (; separated)`.

## Assumptions

* Quantity is used only to decide whether a purchase is valid; similarity is based on
  *whether* someone bought a product, not how much.
* A customer who appears only in rejected rows has no history and is not in the engine.
  This does not happen in the supplied data (all 20 customers appear).
* Product name and category come from the first valid row seen for each product ID.
* In the supplied data most customers share no product with the first five, so the
  similarity matrix for them is mostly 0 off the diagonal; that is a property of the
  data, not a bug (the diagonal is 1.0 and all values are in [0, 1]).

## Layout

```
recommender.py    read_purchases, jaccard, RecommendationEngine
main.py           end-to-end run
data/purchases.csv
output/recommendations.csv
tests/test_recommender.py
requirements.txt
```
