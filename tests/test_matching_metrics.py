"""Tests for common/matching_metrics.py.

Uses a small hand-computable synthetic 1:N matching scenario (2 queries,
3 database entries) so every expected value below was worked out by hand,
not just re-derived from the code under test.

Scenario
--------
query_ids = ["A", "B"], db_ids = ["A", "C", "B"]
score_mat = [[0.9, 0.1, 0.5],   # query A: best score goes to db[0]="A" -> true rank 1
             [0.2, 0.8, 0.3]]   # query B: best score goes to db[1]="C" (wrong) -> true rank 2
"""

import numpy as np
import pandas as pd
import pytest

from common.matching_metrics import (
    build_pair_scores_table,
    build_topk_table,
    build_true_rank_table,
    true_midranks,
)

QUERY_IDS = np.array(["A", "B"])
DB_IDS = np.array(["A", "C", "B"])
SCORE_MAT = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.3]])
DIST_MAT = 1.0 / SCORE_MAT - 1.0  # consistent with score = 1 / (1 + distance)


@pytest.fixture
def query_df():
    return pd.DataFrame({"case_id": ["qA", "qB"], "person_id": ["A", "B"]})


@pytest.fixture
def db_df():
    return pd.DataFrame({"case_id": ["dA", "dC", "dB"], "person_id": ["A", "C", "B"]})


def test_build_true_rank_table_matches_hand_computation(query_df, db_df):
    table = build_true_rank_table(DIST_MAT, SCORE_MAT, query_df, db_df, "case_id", "person_id")
    assert table["true_rank"].tolist() == [1, 2]
    assert table["has_match_in_database"].tolist() == [True, True]
    assert table.loc[0, "matched_db_case"] == "dA"
    assert table.loc[1, "matched_db_case"] == "dB"


def test_build_topk_table_orders_by_score(query_df, db_df):
    table = build_topk_table(DIST_MAT, SCORE_MAT, query_df, db_df, "case_id", "person_id", topk=2)
    query_a_rows = table[table["query_person"] == "A"].sort_values("display_order")
    assert query_a_rows["db_person"].tolist() == ["A", "B"]  # by descending score: 0.9 then 0.5
    assert query_a_rows["correct"].tolist() == [True, False]


def test_build_topk_table_reports_midrank_for_score_ties(query_df, db_df):
    tied_scores = np.asarray([[0.8, 0.8, 0.2], [0.8, 0.3, 0.2]])
    tied_distances = 1.0 / tied_scores - 1.0

    table = build_topk_table(
        tied_distances, tied_scores, query_df, db_df, "case_id", "person_id", topk=3
    )
    query_a = table[table["query_person"].eq("A")].sort_values("display_order")

    assert query_a["display_order"].tolist() == [1, 2, 3]
    assert query_a["score_midrank"].tolist() == [1.5, 1.5, 3.0]
    assert query_a["score_tie_count"].tolist() == [1, 1, 0]


def test_build_pair_scores_table_covers_all_pairs_and_labels(query_df, db_df):
    table = build_pair_scores_table(DIST_MAT, SCORE_MAT, query_df, db_df, "case_id", "person_id")
    assert len(table) == 2 * 3
    genuine = table[table["label"] == 1]
    assert len(genuine) == 2  # (A,A) and (B,B)


def test_true_rank_ties_use_midrank_independent_of_gallery_order():
    scores_a = np.asarray([[0.8, 0.8, 0.2]])
    scores_b = np.asarray([[0.8, 0.2, 0.8]])
    rank_a, ties_a = true_midranks(scores_a, np.asarray(["A"]), np.asarray(["A", "B", "C"]))
    rank_b, ties_b = true_midranks(scores_b, np.asarray(["A"]), np.asarray(["C", "B", "A"]))
    assert rank_a.tolist() == [1.5]
    assert rank_b.tolist() == [1.5]
    assert ties_a.tolist() == ties_b.tolist() == [1]
