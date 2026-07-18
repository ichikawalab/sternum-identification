"""Shared 1:N identity-matching ranks and output tables.

Used by 04_matching/radiomics_matching.py to keep rank and output-table
construction separate from feature preprocessing and distance calculation.

Convention: a higher score means more similar. dist_mat/score_mat are
(n_query, n_db) arrays; query_df/db_df provide case_col and person_col for
each row, in the same order as the score matrix's axes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_midranks(
    score_mat: np.ndarray,
    query_ids: np.ndarray,
    db_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gallery-order-independent genuine mid-ranks and tie counts."""
    n_query = score_mat.shape[0]
    ranks = np.full(n_query, np.nan, dtype=float)
    tie_counts = np.zeros(n_query, dtype=int)
    db_ids = db_ids.astype(str)
    for index, person in enumerate(query_ids.astype(str)):
        genuine = np.flatnonzero(db_ids == person)
        if genuine.size > 1:
            raise ValueError(f"Query identity has multiple genuine references: {person}")
        if genuine.size == 0:
            continue
        genuine_score = score_mat[index, genuine[0]]
        n_better = int(np.sum(score_mat[index] > genuine_score))
        n_tied_other = int(np.sum(score_mat[index] == genuine_score) - 1)
        ranks[index] = 1.0 + n_better + 0.5 * n_tied_other
        tie_counts[index] = n_tied_other
    return ranks, tie_counts


def build_topk_table(
    dist_mat: np.ndarray,
    score_mat: np.ndarray,
    query_df: pd.DataFrame,
    db_df: pd.DataFrame,
    case_col: str,
    person_col: str,
    topk: int,
) -> pd.DataFrame:
    """Build a top-k candidate table for each query."""
    query_cases = query_df[case_col].astype(str).to_numpy()
    query_persons = query_df[person_col].astype(str).to_numpy()
    db_cases = db_df[case_col].astype(str).to_numpy()
    db_persons = db_df[person_col].astype(str).to_numpy()

    rows = []
    for qi in range(score_mat.shape[0]):
        order = np.lexsort((db_cases, -score_mat[qi]))
        for display_order, di in enumerate(order[:topk], start=1):
            n_better = int(np.sum(score_mat[qi] > score_mat[qi, di]))
            n_tied_other = int(np.sum(score_mat[qi] == score_mat[qi, di]) - 1)
            row = {
                "query_case": query_cases[qi],
                "query_person": query_persons[qi],
                "display_order": display_order,
                "score_midrank": 1.0 + n_better + 0.5 * n_tied_other,
                "score_tie_count": n_tied_other,
                "db_case": db_cases[di],
                "db_person": db_persons[di],
                "distance": float(dist_mat[qi, di]),
                "score": float(score_mat[qi, di]),
                "correct": bool(query_persons[qi] == db_persons[di]),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def build_true_rank_table(
    dist_mat: np.ndarray,
    score_mat: np.ndarray,
    query_df: pd.DataFrame,
    db_df: pd.DataFrame,
    case_col: str,
    person_col: str,
) -> pd.DataFrame:
    """Build the genuine-match table using gallery-order-independent mid-ranks."""
    query_cases = query_df[case_col].astype(str).to_numpy()
    query_ids = query_df[person_col].astype(str).to_numpy()
    db_cases = db_df[case_col].astype(str).to_numpy()
    db_ids = db_df[person_col].astype(str).to_numpy()

    true_rank, tie_counts = true_midranks(score_mat, query_ids, db_ids)
    rows = []
    for qi in range(score_mat.shape[0]):
        genuine = np.flatnonzero(db_ids == query_ids[qi])
        if genuine.size:
            di = int(genuine[0])
            rows.append(
                {
                    "query_case": query_cases[qi],
                    "query_person": query_ids[qi],
                    "has_match_in_database": True,
                    "true_rank": float(true_rank[qi]),
                    "genuine_tie_count": int(tie_counts[qi]),
                    "has_score_tie": bool(tie_counts[qi] > 0),
                    "rank_policy": "midrank",
                    "true_distance": float(dist_mat[qi, di]),
                    "true_score": float(score_mat[qi, di]),
                    "matched_db_case": db_cases[di],
                    "matched_db_person": db_ids[di],
                }
            )
        else:
            rows.append(
                {
                    "query_case": query_cases[qi],
                    "query_person": query_ids[qi],
                    "has_match_in_database": False,
                    "true_rank": np.nan,
                    "genuine_tie_count": 0,
                    "has_score_tie": False,
                    "rank_policy": "midrank",
                    "true_distance": np.nan,
                    "true_score": np.nan,
                    "matched_db_case": "",
                    "matched_db_person": "",
                }
            )
    return pd.DataFrame(rows)


def build_pair_scores_table(
    dist_mat: np.ndarray,
    score_mat: np.ndarray,
    query_df: pd.DataFrame,
    db_df: pd.DataFrame,
    case_col: str,
    person_col: str,
) -> pd.DataFrame:
    """Build a raw data table for every query-database pair (all n_query x n_db rows),
    so ROC curves, distance distributions, PR curves, etc. can be regenerated later."""
    query_cases = query_df[case_col].astype(str).to_numpy()
    query_persons = query_df[person_col].astype(str).to_numpy()
    db_cases = db_df[case_col].astype(str).to_numpy()
    db_persons = db_df[person_col].astype(str).to_numpy()

    rows = []
    for qi in range(len(query_cases)):
        for di in range(len(db_cases)):
            row = {
                "query_case": query_cases[qi],
                "query_person": query_persons[qi],
                "db_case": db_cases[di],
                "db_person": db_persons[di],
                "label": int(query_persons[qi] == db_persons[di]),
                "distance": float(dist_mat[qi, di]),
                "score": float(score_mat[qi, di]),
            }
            rows.append(row)
    return pd.DataFrame(rows)
