from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import common.efa_scoring as EFA_SCORING
from common.efa_scoring import fused_score_matrix, true_ranks, view_score_matrix
from common.provenance import (
    require_manifest_output,
    resolve_worker_count,
    safe_file_reference,
    sha256_file,
)
from common.schemas import (
    attach_feature_table,
    dataset_from_person_id,
    radiomics_shape_columns,
    validate_case_metadata,
    validate_matching_cohorts,
)


def load_script(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COHORTS = load_script("build_cohorts_test_module", "03_quality_control/build_cohorts.py")
CROSSFIT = load_script(
    "crossfit_test_module",
    "04_matching/crossfit_efa_matching.py",
)
RADIOMICS_MATCHING = load_script(
    "radiomics_matching_test_module",
    "04_matching/radiomics_matching.py",
)
STATS_COMMON = load_script(
    "_common",
    "05_statistics/_common.py",
)
AUC_INFERENCE = load_script(
    "auc_inference_test_module",
    "05_statistics/auc_inference.py",
)
QC = load_script("mahalanobis_qc_test_module", "03_quality_control/mahalanobis_qc.py")
VISUAL_REVIEW = load_script(
    "visual_case_review_test_module",
    "03_quality_control/visual_case_review.py",
)


def test_validate_case_metadata_rejects_duplicate_case_id() -> None:
    frame = pd.DataFrame(
        {
            "case_id": ["A", "A"],
            "person_id": ["P1", "P2"],
            "pre_0_post_1": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="case_id must be unique"):
        validate_case_metadata(frame, "test")


def test_manifest_validation_rejects_changed_output(tmp_path: Path) -> None:
    output = tmp_path / "table.csv"
    output.write_text("a\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "completed": True,
                "output": {"sha256": sha256_file(output)},
            }
        ),
        encoding="utf-8",
    )
    output.write_text("a\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        require_manifest_output(manifest, output, ("output", "sha256"))


@pytest.mark.parametrize("requested", [0, -2])
def test_worker_count_rejects_invalid_values(requested: int) -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        resolve_worker_count(requested)


def test_dataset_prefix_contract_is_explicit() -> None:
    assert dataset_from_person_id("INST_PAIR_001") == "institutional"
    assert dataset_from_person_id("LIDC-IDRI-0001") == "lidc"
    with pytest.raises(ValueError, match="Unrecognized"):
        dataset_from_person_id("unknown")


def test_attach_feature_table_rejects_failed_locked_case() -> None:
    cohort = pd.DataFrame({"case_id": ["A"], "person_id": ["INST_PAIR_001"], "pre_0_post_1": [1]})
    features = cohort.assign(status="failed", error_message="test", feature_1=1.0)
    with pytest.raises(ValueError, match="failed locked cases"):
        attach_feature_table(cohort, features, "method")


def test_radiomics_shape_contract_accepts_canonical_table() -> None:
    canonical = ["case_id", "original_shape_Elongation", "original_shape_Flatness"]
    assert radiomics_shape_columns(canonical) == canonical[1:]


def test_radiomics_shape_contract_rejects_legacy_only_table() -> None:
    with pytest.raises(ValueError, match="No radiomics"):
        radiomics_shape_columns(["116_original_shape_Elongation"])


def test_matching_contract_rejects_pre_query() -> None:
    query = pd.DataFrame(
        {
            "case_id": ["A_PRE"],
            "person_id": ["A"],
            "pre_0_post_1": [0],
            "cohort_policy": ["primary"],
        }
    )
    reference = pd.DataFrame(
        {
            "case_id": ["A_REF"],
            "person_id": ["A"],
            "pre_0_post_1": [0],
            "cohort_policy": ["primary"],
        }
    )
    with pytest.raises(ValueError, match="POST-only"):
        validate_matching_cohorts(query, reference)


def test_matching_contract_rejects_mixed_cohort_policies() -> None:
    query = pd.DataFrame(
        {
            "case_id": ["A_POST"],
            "person_id": ["A"],
            "pre_0_post_1": [1],
            "cohort_policy": ["technical_only_sensitivity"],
        }
    )
    reference = pd.DataFrame(
        {
            "case_id": ["A_PRE", "B_PRE"],
            "person_id": ["A", "B"],
            "pre_0_post_1": [0, 0],
            "cohort_policy": ["primary", "primary"],
        }
    )

    with pytest.raises(ValueError, match="same single cohort_policy"):
        validate_matching_cohorts(query, reference)


def test_matching_contract_rejects_missing_cohort_policy_rows() -> None:
    query = pd.DataFrame(
        {
            "case_id": ["A_POST", "B_POST"],
            "person_id": ["A", "B"],
            "pre_0_post_1": [1, 1],
            "cohort_policy": ["primary", pd.NA],
        }
    )
    reference = pd.DataFrame(
        {
            "case_id": ["A_PRE", "B_PRE"],
            "person_id": ["A", "B"],
            "pre_0_post_1": [0, 0],
            "cohort_policy": ["primary", "primary"],
        }
    )

    with pytest.raises(ValueError, match="non-empty for every"):
        validate_matching_cohorts(query, reference)


def test_radiomics_scaler_excludes_evaluated_identity() -> None:
    query = pd.DataFrame({"feature": [5.0]})
    reference = pd.DataFrame({"feature": [100.0, 0.0, 10.0]})
    distance, score = RADIOMICS_MATCHING.compute_leave_one_person_out_scores(
        query,
        reference,
        np.asarray(["A"]),
        np.asarray(["A", "B", "C"]),
    )

    assert distance[0].tolist() == pytest.approx([9.5, 0.5, 0.5])
    assert score[0, 0] == pytest.approx(1.0 / 10.5)


def test_common_hard_failure_includes_either_feature_method() -> None:
    frame = pd.DataFrame(
        {
            "status": ["success", "success"],
            "efa_status": ["success", "failed"],
        }
    )
    features = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    efa = pd.DataFrame({"cor_H5_a1": [1.0, 2.0]})
    hard, reason = QC.hard_failure_flags(frame, features, efa)
    assert hard.tolist() == [False, True]
    assert reason.tolist() == ["", "efa_failure"]


def test_common_hard_failure_rejects_nonfinite_efa_coefficients() -> None:
    frame = pd.DataFrame({"status": ["success"], "efa_status": ["success"]})
    radiomics = pd.DataFrame({"f1": [1.0]})
    efa = pd.DataFrame({"cor_H5_a1": [np.inf]})

    hard, reason = QC.hard_failure_flags(frame, radiomics, efa)

    assert hard.tolist() == [True]
    assert reason.tolist() == ["non_finite_efa_feature"]


def test_mahalanobis_is_fitted_separately_by_dataset() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["institutional"] * 12 + ["lidc"] * 12,
        }
    )
    features = pd.DataFrame(
        {
            "f1": list(range(12)) + list(range(100, 112)),
            "f2": [value * value for value in range(12)] + [value * 3 for value in range(12)],
        }
    )
    hard = pd.Series(False, index=frame.index)
    out, manifests = QC.apply_dataset_qc(frame, features, hard, 0.95)
    assert [item["dataset"] for item in manifests] == ["institutional", "lidc"]
    assert all(item["n_mahalanobis_fit"] == 12 for item in manifests)
    assert out.groupby("dataset")["mahalanobis_distance_squared"].count().to_dict() == {
        "institutional": 12,
        "lidc": 12,
    }


def synthetic_qc_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for number in range(1, 67):
        person = f"INST_PAIR_{number:03d}"
        for flag in (0, 1):
            rows.append(
                {
                    "case_id": f"{person}_{'POST' if flag else 'PRE'}",
                    "person_id": person,
                    "pre_0_post_1": flag,
                    "hard_qc_failure": False,
                    "hard_qc_reason": "",
                    "mahalanobis_distance_squared": 1.0,
                    "mahalanobis_threshold": 10.0,
                    "mahalanobis_outlier": False,
                    "include_technical_only": True,
                    "include_mahalanobis": True,
                }
            )
    lidc_people = [f"LIDC-IDRI-{number:04d}" for number in range(1, 1007)]
    lidc_people.extend(lidc_people[:8])
    for index, person in enumerate(lidc_people, start=1):
        rows.append(
            {
                "case_id": f"{person}_S{index:04d}",
                "person_id": person,
                "pre_0_post_1": 0,
                "hard_qc_failure": False,
                "hard_qc_reason": "",
                "mahalanobis_distance_squared": 1.0,
                "mahalanobis_threshold": 10.0,
                "mahalanobis_outlier": False,
                "include_technical_only": True,
                "include_mahalanobis": True,
            }
        )
    return pd.DataFrame(rows)


def test_minimal_sensitivity_cohorts_change_one_factor_at_a_time() -> None:
    frame = synthetic_qc_table()
    frame.loc[frame["case_id"].eq("INST_PAIR_001_PRE"), "mahalanobis_outlier"] = True
    frame.loc[frame["case_id"].eq("INST_PAIR_001_PRE"), "include_mahalanobis"] = False
    checked = COHORTS.validate_qc_table(frame)

    primary = COHORTS.build_policy_cohort(checked, "primary", "include_mahalanobis", False)
    technical_only = COHORTS.build_policy_cohort(
        checked, "technical_only_sensitivity", "include_technical_only", False
    )
    one_per_person = COHORTS.build_policy_cohort(
        checked, "lidc_one_per_person", "include_mahalanobis", True
    )

    assert primary[3]["n_query"] == 65
    assert primary[3]["n_lidc_reference"] == 1014
    assert technical_only[3]["n_query"] == 66
    assert technical_only[3]["n_lidc_reference"] == 1014
    assert one_per_person[3]["n_query"] == 65
    assert one_per_person[3]["n_lidc_reference"] == 1006


def test_cohort_validation_rejects_inconsistent_inclusion_flags() -> None:
    frame = synthetic_qc_table()
    frame.loc[frame["case_id"].eq("INST_PAIR_001_PRE"), "include_technical_only"] = False

    with pytest.raises(ValueError, match="include_technical_only disagrees"):
        COHORTS.validate_qc_table(frame)


def test_stage03_synthetic_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = synthetic_qc_table()[["case_id", "person_id", "pre_0_post_1"]].copy()
    index = np.arange(len(metadata), dtype=float)
    radiomics = metadata.assign(
        status="success",
        error_message="",
        original_shape_feature1=index,
        original_shape_feature2=(index % 17.0) ** 2,
    )
    efa = metadata.assign(
        status="success",
        error_message="",
        cor_H5_a1=index / 10.0,
    )
    radiomics_path = tmp_path / "radiomics.csv"
    efa_path = tmp_path / "efa.csv"
    qc_path = tmp_path / "qc.csv"
    cohort_root = tmp_path / "cohorts"
    radiomics.to_csv(radiomics_path, index=False)
    efa.to_csv(efa_path, index=False)
    segmentation_reference = {"name": "run_manifest.json", "sha256": "synthetic"}
    radiomics_path.with_suffix(".run_manifest.json").write_text(
        json.dumps(
            {
                "completed": True,
                "output_csv": {
                    "name": radiomics_path.name,
                    "sha256": sha256_file(radiomics_path),
                },
                "segmentation_manifest": segmentation_reference,
            }
        ),
        encoding="utf-8",
    )
    (efa_path.parent / "efa_run_manifest.json").write_text(
        json.dumps(
            {
                "completed": True,
                "outputs": {efa_path.name: sha256_file(efa_path)},
                "segmentation_manifest": segmentation_reference,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mahalanobis_qc.py",
            "--input_csv",
            str(radiomics_path),
            "--efa_features_csv",
            str(efa_path),
            "--output_csv",
            str(qc_path),
        ],
    )
    QC.main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_cohorts.py", "--qc_csv", str(qc_path), "--out_dir", str(cohort_root)],
    )
    COHORTS.main()

    qc = pd.read_csv(qc_path)
    primary_query = pd.read_csv(cohort_root / "primary" / "query.csv")
    assert len(qc) == 1146
    institution = qc[qc["person_id"].str.startswith("INST_PAIR_")]
    expected_query = int(institution.groupby("person_id")["include_mahalanobis"].all().sum())
    assert len(primary_query) == expected_query
    assert (tmp_path / "qc.manifest.json").is_file()
    assert (cohort_root / "primary" / "manifest.json").is_file()


def test_shared_efa_scoring_recovers_genuine_reference() -> None:
    reference = pd.DataFrame(
        {
            "person_id": ["A", "B", "C"],
            "cor_H1_a1": [0.0, 5.0, 10.0],
            "sag_H1_a1": [0.0, 5.0, 10.0],
            "axial_H1_a1": [0.0, 5.0, 10.0],
        }
    )
    query = pd.DataFrame(
        {
            "person_id": ["A", "B"],
            "cor_H1_a1": [0.1, 4.9],
            "sag_H1_a1": [0.1, 4.9],
            "axial_H1_a1": [0.1, 4.9],
        }
    )
    _, scores, count, kind = fused_score_matrix(query, reference, "cor_sag_axial", harmonics=1)
    ranks = true_ranks(
        scores,
        query["person_id"].to_numpy(),
        reference["person_id"].to_numpy(),
    )
    assert count == 3
    assert kind == "equivalent_distance"
    assert ranks.tolist() == [1.0, 1.0]


def test_crossfit_tie_break_prefers_simpler_configuration() -> None:
    ranks = pd.Series([1.0, 1.0, 2.0]).to_numpy()
    simple = CROSSFIT.selection_key(ranks, "cor_sag", 10)
    complex_key = CROSSFIT.selection_key(ranks, "cor_sag_axial", 20)
    assert simple < complex_key


def test_crossfit_secondary_rule_is_mean_log_true_rank() -> None:
    ranks = np.asarray([1.0, 2.0, 4.0])

    key = CROSSFIT.selection_key(ranks, "cor", 5)

    assert key[1] == pytest.approx(np.mean(np.log(ranks)))


def test_crossfit_candidate_grid_is_locked_to_28_configurations() -> None:
    assert len(CROSSFIT.CANDIDATE_CONFIGURATIONS) == 28
    assert len(set(CROSSFIT.CANDIDATE_CONFIGURATIONS)) == 28
    assert set(CROSSFIT.CANDIDATE_HARMONICS) == {5, 10, 20, 30}


def test_crossfit_analysis_role_requires_all_primary_settings() -> None:
    assert (
        CROSSFIT.analysis_role("primary", "area_normalized", CROSSFIT.CANDIDATE_CONFIGURATIONS)
        == "primary"
    )
    assert (
        CROSSFIT.analysis_role("primary", "size_preserved", CROSSFIT.CANDIDATE_CONFIGURATIONS)
        == "sensitivity"
    )
    assert CROSSFIT.analysis_role("primary", "area_normalized", (("cor_sag", 20),)) == "sensitivity"


def test_crossfit_cli_rejects_representation_filename_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crossfit_efa_matching.py",
            "--query_csv",
            "query.csv",
            "--reference_csv",
            "reference.csv",
            "--features_csv",
            "efa_features_size_preserved.csv",
            "--feature_representation",
            "area_normalized",
            "--out_dir",
            "out",
        ],
    )

    with pytest.raises(ValueError, match="filename disagrees"):
        CROSSFIT.main()


def test_view_scaling_can_exclude_held_out_reference() -> None:
    fit_reference = pd.DataFrame({"cor_H1_a1": [0.0, 10.0]})
    evaluation_reference = pd.DataFrame({"cor_H1_a1": [0.0, 10.0, 100.0]})
    query = pd.DataFrame({"cor_H1_a1": [10.0]})

    distance, _, _ = view_score_matrix(
        query,
        evaluation_reference,
        "cor",
        harmonics=1,
        fit_reference=fit_reference,
    )

    assert distance[0].tolist() == pytest.approx([1.0, 0.0, 9.0])


def test_fused_scoring_cache_computes_each_view_once(monkeypatch) -> None:
    query = pd.DataFrame({"cor_H1_a1": [0.0], "sag_H1_a1": [0.0], "axial_H1_a1": [0.0]})
    reference = pd.DataFrame(
        {"cor_H1_a1": [0.0, 1.0], "sag_H1_a1": [0.0, 1.0], "axial_H1_a1": [0.0, 1.0]}
    )
    original = EFA_SCORING.view_score_matrix
    calls: list[str] = []

    def counted(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(EFA_SCORING, "view_score_matrix", counted)
    cache = {}
    for mode in EFA_SCORING.MODE_ORDER:
        EFA_SCORING.fused_score_matrix(query, reference, mode, harmonics=1, scoring_cache=cache)
    assert sorted(calls) == ["axial", "cor", "sag"]


def test_crossfit_excludes_both_scans_from_selection_and_scaler(monkeypatch) -> None:
    query = pd.DataFrame(
        {
            "case_id": ["A_POST", "B_POST"],
            "person_id": ["A", "B"],
            "pre_0_post_1": [1, 1],
            "cohort_policy": ["primary", "primary"],
        }
    )
    reference = pd.DataFrame(
        {
            "case_id": ["A_PRE", "B_PRE", "LIDC_1"],
            "person_id": ["A", "B", "LIDC"],
            "pre_0_post_1": [0, 0, 0],
            "cohort_policy": ["primary", "primary", "primary"],
        }
    )
    evaluation_calls: list[tuple[str, set[str], set[str]]] = []

    def fake_fused_score_matrix(
        query_frame,
        reference_frame,
        mode,
        harmonics,
        fit_reference=None,
        scoring_cache=None,
    ):
        del mode, harmonics, scoring_cache
        query_people = query_frame["person_id"].astype(str).to_numpy()
        reference_people = reference_frame["person_id"].astype(str).to_numpy()
        fit_people = set(fit_reference["person_id"].astype(str))
        score = (query_people[:, None] == reference_people[None, :]).astype(float)
        if len(reference_frame) == 3:
            evaluation_calls.append((query_people[0], set(reference_people), fit_people))
        return 1.0 - score, score, 1, "distance"

    monkeypatch.setattr(CROSSFIT, "fused_score_matrix", fake_fused_score_matrix)
    held_out, pair_scores, selection_audit = CROSSFIT.run_crossfit(query, reference)

    assert held_out["true_rank"].tolist() == [1.0, 1.0]
    assert len(pair_scores) == 6
    assert len(selection_audit) == 2 * len(CROSSFIT.CANDIDATE_CONFIGURATIONS)
    assert selection_audit.groupby("held_out_person")["selected"].sum().eq(1).all()
    assert pair_scores.groupby("query_person")["label"].sum().to_dict() == {"A": 1, "B": 1}
    assert len(evaluation_calls) == 2
    for held_out_person, gallery_people, fit_people in evaluation_calls:
        assert held_out_person in gallery_people
        assert held_out_person not in fit_people


def test_pair_loader_preserves_database_case_for_paired_auc(tmp_path) -> None:
    path = tmp_path / "pairs.csv"
    pd.DataFrame(
        {
            "query_person": ["A", "A"],
            "db_case": ["A_PRE", "B_PRE"],
            "label": [1, 0],
            "score": [0.9, 0.1],
        }
    ).to_csv(path, index=False)

    loaded = STATS_COMMON.load_pairs(path)

    assert loaded.columns.tolist() == ["query_person", "db_case", "label", "score"]
    assert loaded["db_case"].tolist() == ["A_PRE", "B_PRE"]


def test_statistics_loaders_reject_invalid_identifiers_and_scores(tmp_path) -> None:
    rank_path = tmp_path / "rank.csv"
    pd.DataFrame({"query_person": ["A", "A"], "true_rank": [1, 2]}).to_csv(rank_path, index=False)
    with pytest.raises(ValueError, match="one row per query person"):
        STATS_COMMON.load_true_rank(rank_path)

    pairs_path = tmp_path / "pairs.csv"
    pd.DataFrame(
        {
            "query_person": ["A", "A"],
            "db_case": ["A_PRE", "B_PRE"],
            "label": [1, 0],
            "score": ["invalid", 0.1],
        }
    ).to_csv(pairs_path, index=False)
    with pytest.raises(ValueError, match="non-numeric labels or scores"):
        STATS_COMMON.load_pairs(pairs_path)

    pd.DataFrame(
        {
            "query_person": ["A", "A"],
            "db_case": ["A_PRE", "B_PRE"],
            "label": [1, 0.5],
            "score": [0.9, 0.1],
        }
    ).to_csv(pairs_path, index=False)
    with pytest.raises(ValueError, match="labels other than 0 and 1"):
        STATS_COMMON.load_pairs(pairs_path)


def test_statistics_requires_matching_outputs_from_same_locked_cohort(tmp_path) -> None:
    shared_cohort = {"name": "manifest.json", "sha256": "cohort-hash"}
    paths = []
    for method in ("a", "b"):
        directory = tmp_path / method
        directory.mkdir()
        output = directory / "true_rank.csv"
        output.write_text("query_person,true_rank\nP1,1\n", encoding="utf-8")
        manifest = {
            "completed": True,
            "cohort_policy": "primary",
            "n_query": 1,
            "n_reference": 2,
            "inputs": {
                "query_csv": {"name": "query.csv", "sha256": "query-hash"},
                "reference_csv": {
                    "name": "reference_gallery.csv",
                    "sha256": "reference-hash",
                },
            },
            "upstream": {"cohort_manifest": shared_cohort},
            "outputs": {output.name: sha256_file(output)},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(output)

    provenance = STATS_COMMON.require_paired_matching_outputs(paths[0], paths[1])

    assert provenance["cohort_policy"] == "primary"


def test_auc_score_assigns_half_credit_to_exact_ties() -> None:
    assert STATS_COMMON.auc_score(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_query_macro_auc_is_invariant_to_between_query_score_scale() -> None:
    groups = {
        "A": pd.DataFrame({"label": [1, 0, 0], "score": [0.9, 0.2, 0.1]}),
        "B": pd.DataFrame({"label": [1, 0, 0], "score": [900.0, 200.0, 100.0]}),
    }

    auc_values = AUC_INFERENCE.per_query_metrics(groups, ["A", "B"])

    assert auc_values.tolist() == [1.0, 1.0]


def test_auc_sign_flip_test_is_reproducible() -> None:
    differences = np.asarray([0.2, 0.1, -0.05, 0.3])
    first = AUC_INFERENCE.paired_sign_flip_pvalue(differences, 1000, 42)
    second = AUC_INFERENCE.paired_sign_flip_pvalue(differences, 1000, 42)
    assert first == second
    assert 0.0 < first <= 1.0
    assert AUC_INFERENCE.paired_sign_flip_pvalue(np.zeros(4), 10, 42) == 1.0


def test_visual_rank1_cross_tab_excludes_not_evaluated_people() -> None:
    review = pd.DataFrame(
        {
            "person_id": ["INST_PAIR_001", "INST_PAIR_002", "INST_PAIR_003"],
            "pre_segmentation_quality": ["acceptable", "major_error", "major_error"],
            "post_segmentation_quality": ["acceptable", "acceptable", "acceptable"],
            "pre_coverage": ["complete", "complete", "partial"],
            "post_coverage": ["complete", "complete", "complete"],
            "fracture_or_deformity": ["absent", "present", "present"],
            "degenerative_change": ["absent", "absent", "present"],
            "pose_issue": ["acceptable", "questionable", "questionable"],
        }
    )
    rank = pd.DataFrame({"person_id": ["INST_PAIR_001", "INST_PAIR_002"], "true_rank": [1, 2]})
    summary = VISUAL_REVIEW.summarize_visual_findings_by_rank1(review, rank)
    assert summary["n_reviewed_total"].eq(3).all()
    assert summary["n_evaluated_primary"].eq(2).all()
    assert summary["n_not_evaluated_primary"].eq(1).all()
    failure_major = summary[
        summary["rank_group"].eq("not_rank_1")
        & summary["finding"].eq("any_major_segmentation_error")
    ].iloc[0]
    assert failure_major["n_group"] == 1
    assert failure_major["n_finding"] == 1


def test_visual_review_summarizes_mahalanobis_relation() -> None:
    review = pd.DataFrame(
        {
            "person_id": ["INST_PAIR_001", "INST_PAIR_002"],
            "pre_segmentation_quality": ["major_error", "acceptable"],
            "post_segmentation_quality": ["acceptable", "minor_error"],
            "fracture_or_deformity": ["present", "absent"],
            "pose_issue": ["questionable", "acceptable"],
        }
    )
    cohort = pd.DataFrame(
        {
            "case_id": [
                "INST_PAIR_001_PRE",
                "INST_PAIR_001_POST",
                "INST_PAIR_002_PRE",
                "INST_PAIR_002_POST",
            ],
            "person_id": ["INST_PAIR_001", "INST_PAIR_001", "INST_PAIR_002", "INST_PAIR_002"],
            "pre_0_post_1": [0, 1, 0, 1],
            "mahalanobis_outlier": [False, True, False, False],
        }
    )
    summary = VISUAL_REVIEW.summarize_mahalanobis_relation(review, cohort)
    outlier = summary.set_index("mahalanobis_group").loc["outlier"]
    assert outlier["n_people"] == 1
    assert outlier["n_any_major_segmentation_error"] == 1
    assert outlier["n_visually_apparent_fracture_or_deformity"] == 1


def test_visual_review_template_masks_current_matching_outcomes() -> None:
    cohort = pd.DataFrame(
        {
            "case_id": ["INST_PAIR_001_PRE", "INST_PAIR_001_POST"],
            "person_id": ["INST_PAIR_001", "INST_PAIR_001"],
            "pre_0_post_1": [0, 1],
            "include_technical_only": [True, True],
            "selected_for_matching": [True, True],
            "matching_role": ["true_reference", "query"],
        }
    )
    template = VISUAL_REVIEW.create_outcome_masked_template(cohort)
    assert "true_rank" not in template.columns
    assert "failure_category" not in template.columns
    assert "degenerative_change" in template.columns


def test_visual_review_template_excludes_unselected_pairs() -> None:
    cohort = pd.DataFrame(
        {
            "case_id": [
                "INST_PAIR_001_PRE",
                "INST_PAIR_001_POST",
                "INST_PAIR_002_PRE",
                "INST_PAIR_002_POST",
            ],
            "person_id": ["INST_PAIR_001", "INST_PAIR_001", "INST_PAIR_002", "INST_PAIR_002"],
            "pre_0_post_1": [0, 1, 0, 1],
            "include_technical_only": [True, True, False, False],
            "selected_for_matching": [True, True, False, False],
            "matching_role": ["true_reference", "query", "excluded", "excluded"],
        }
    )

    template = VISUAL_REVIEW.create_outcome_masked_template(cohort)

    assert template["person_id"].tolist() == ["INST_PAIR_001"]


def test_visual_review_template_drops_one_sided_technical_failure() -> None:
    cohort = pd.DataFrame(
        {
            "case_id": [
                "INST_PAIR_001_PRE",
                "INST_PAIR_001_POST",
                "INST_PAIR_018_PRE",
                "INST_PAIR_018_POST",
            ],
            "person_id": ["INST_PAIR_001", "INST_PAIR_001", "INST_PAIR_018", "INST_PAIR_018"],
            "pre_0_post_1": [0, 1, 0, 1],
            "include_technical_only": [True, True, False, True],
            "selected_for_matching": [True, True, False, False],
            "matching_role": ["true_reference", "query", "excluded", "excluded"],
        }
    )

    template = VISUAL_REVIEW.create_outcome_masked_template(cohort)

    assert template["person_id"].tolist() == ["INST_PAIR_001"]


def test_visual_review_rejects_case_id_changed_after_template_creation() -> None:
    cohort = pd.DataFrame(
        {
            "case_id": ["INST_PAIR_001_PRE", "INST_PAIR_001_POST"],
            "person_id": ["INST_PAIR_001", "INST_PAIR_001"],
            "pre_0_post_1": [0, 1],
            "include_technical_only": [True, True],
            "selected_for_matching": [True, True],
            "matching_role": ["true_reference", "query"],
        }
    )
    review = VISUAL_REVIEW.create_outcome_masked_template(cohort)
    review.loc[0, "pre_case_id"] = "WRONG_PRE"
    review.loc[0, "reviewer_id"] = "reviewer_1"
    review.loc[0, "review_date"] = "2026-07-13"
    for field in VISUAL_REVIEW.OUTCOME_MASKED_REVIEW_FIELDS:
        review.loc[0, field] = sorted(VISUAL_REVIEW.OUTCOME_MASKED_REVIEW_FIELDS[field])[0]

    with pytest.raises(ValueError, match="pre_case_id disagrees"):
        VISUAL_REVIEW.validate_outcome_masked_review(review, cohort)


def test_visual_review_rejects_outcome_or_free_text_columns() -> None:
    cohort = pd.DataFrame(
        {
            "case_id": ["INST_PAIR_001_PRE", "INST_PAIR_001_POST"],
            "person_id": ["INST_PAIR_001", "INST_PAIR_001"],
            "pre_0_post_1": [0, 1],
            "include_technical_only": [True, True],
        }
    )
    review = VISUAL_REVIEW.create_outcome_masked_template(cohort)
    review["true_rank"] = 1

    with pytest.raises(ValueError, match="unsupported columns.*true_rank"):
        VISUAL_REVIEW.validate_outcome_masked_review(review, cohort)


def test_visual_review_verifies_cohort_and_rank_manifests(tmp_path: Path) -> None:
    cohort_dir = tmp_path / "cohort"
    rank_dir = tmp_path / "matching"
    cohort_dir.mkdir()
    rank_dir.mkdir()
    cohort_path = cohort_dir / "cohort_audit.csv"
    cohort = pd.DataFrame(
        {
            "case_id": ["INST_PAIR_001_PRE", "INST_PAIR_001_POST"],
            "person_id": ["INST_PAIR_001", "INST_PAIR_001"],
            "pre_0_post_1": [0, 1],
            "include_technical_only": [True, True],
            "selected_for_matching": [True, True],
            "matching_role": ["true_reference", "query"],
        }
    )
    cohort.to_csv(cohort_path, index=False)
    cohort_manifest_path = cohort_dir / "manifest.json"
    cohort_manifest_path.write_text(
        json.dumps(
            {
                "pipeline": "sternum_cohort_locking",
                "completed": True,
                "outputs": {cohort_path.name: sha256_file(cohort_path)},
            }
        ),
        encoding="utf-8",
    )
    loaded, _, cohort_reference, _ = VISUAL_REVIEW.load_verified_cohort(cohort_path)
    rank_path = rank_dir / "crossfit_true_rank.csv"
    pd.DataFrame({"query_person": ["INST_PAIR_001"], "true_rank": [1.0]}).to_csv(
        rank_path, index=False
    )
    (rank_dir / "manifest.json").write_text(
        json.dumps(
            {
                "completed": True,
                "analysis_role": "primary",
                "n_query": 1,
                "upstream": {"cohort_manifest": cohort_reference},
                "outputs": {rank_path.name: sha256_file(rank_path)},
            }
        ),
        encoding="utf-8",
    )

    rank, _, _ = VISUAL_REVIEW.load_verified_primary_rank(
        rank_path,
        loaded,
        cohort_reference,
    )
    assert rank["person_id"].tolist() == ["INST_PAIR_001"]

    rank_path.write_text("query_person,true_rank\nINST_PAIR_001,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        VISUAL_REVIEW.load_verified_primary_rank(rank_path, loaded, cohort_reference)


def test_crossfit_runs_all_locked_candidates_with_real_scoring() -> None:
    people = ["A", "B", "C"]
    query = pd.DataFrame(
        {
            "case_id": [f"{person}_POST" for person in people],
            "person_id": people,
            "pre_0_post_1": [1, 1, 1],
            "cohort_policy": ["primary"] * 3,
        }
    )
    reference = pd.DataFrame(
        {
            "case_id": ["A_PRE", "B_PRE", "C_PRE", "LIDC_1"],
            "person_id": ["A", "B", "C", "LIDC"],
            "pre_0_post_1": [0, 0, 0, 0],
            "cohort_policy": ["primary"] * 4,
        }
    )
    base_reference = [0.0, 10.0, 20.0, 100.0]
    base_query = [0.1, 10.1, 20.1]
    for view in ("cor", "sag", "axial"):
        for harmonic in CROSSFIT.CANDIDATE_HARMONICS:
            column = f"{view}_H{harmonic}_a1"
            reference[column] = base_reference
            query[column] = base_query

    held_out, pair_scores, selection_audit = CROSSFIT.run_crossfit(query, reference)

    assert held_out["true_rank"].tolist() == [1.0, 1.0, 1.0]
    assert held_out["selected_mode"].tolist() == ["cor", "cor", "cor"]
    assert held_out["selected_harmonic"].tolist() == [5, 5, 5]
    assert len(pair_scores) == 12
    assert len(selection_audit) == 3 * len(CROSSFIT.CANDIDATE_CONFIGURATIONS)


def write_small_matching_inputs(tmp_path: Path) -> dict[str, Path]:
    people = ["A", "B", "C"]
    query = pd.DataFrame(
        {
            "case_id": [f"{person}_POST" for person in people],
            "person_id": people,
            "pre_0_post_1": [1, 1, 1],
            "cohort_policy": ["primary"] * 3,
        }
    )
    reference = pd.DataFrame(
        {
            "case_id": ["A_PRE", "B_PRE", "C_PRE", "LIDC_1"],
            "person_id": ["A", "B", "C", "LIDC"],
            "pre_0_post_1": [0, 0, 0, 0],
            "cohort_policy": ["primary"] * 4,
        }
    )
    metadata = pd.concat(
        [
            query[["case_id", "person_id", "pre_0_post_1"]],
            reference[["case_id", "person_id", "pre_0_post_1"]],
        ],
        ignore_index=True,
    )
    values = np.asarray([0.1, 10.1, 20.1, 0.0, 10.0, 20.0, 100.0])
    radiomics = metadata.assign(
        status="success",
        error_message="",
        original_shape_feature1=values,
        original_shape_feature2=values**2,
    )
    efa = metadata.assign(status="success", error_message="")
    for view in ("cor", "sag", "axial"):
        for harmonic in CROSSFIT.CANDIDATE_HARMONICS:
            efa[f"{view}_H{harmonic}_a1"] = values

    paths = {
        "query": tmp_path / "query.csv",
        "reference": tmp_path / "reference.csv",
        "radiomics": tmp_path / "radiomics.csv",
        "efa": tmp_path / "efa_features_area_normalized.csv",
    }
    query.to_csv(paths["query"], index=False)
    reference.to_csv(paths["reference"], index=False)
    radiomics.to_csv(paths["radiomics"], index=False)
    efa.to_csv(paths["efa"], index=False)
    segmentation_reference = {"name": "run_manifest.json", "sha256": "synthetic"}
    radiomics_manifest_path = paths["radiomics"].with_suffix(".run_manifest.json")
    efa_manifest_path = tmp_path / "efa_run_manifest.json"
    radiomics_manifest_path.write_text(
        json.dumps(
            {
                "completed": True,
                "output_csv": {
                    "name": paths["radiomics"].name,
                    "sha256": sha256_file(paths["radiomics"]),
                },
                "segmentation_manifest": segmentation_reference,
            }
        ),
        encoding="utf-8",
    )
    efa_manifest_path.write_text(
        json.dumps(
            {
                "completed": True,
                "outputs": {paths["efa"].name: sha256_file(paths["efa"])},
                "segmentation_manifest": segmentation_reference,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "pipeline": "sternum_cohort_locking",
                "completed": True,
                "outputs": {
                    paths["query"].name: sha256_file(paths["query"]),
                    paths["reference"].name: sha256_file(paths["reference"]),
                },
                "qc_feature_manifests": {
                    "radiomics": safe_file_reference(radiomics_manifest_path),
                    "efa": safe_file_reference(efa_manifest_path),
                },
                "segmentation_manifest": segmentation_reference,
            }
        ),
        encoding="utf-8",
    )
    return paths


def test_matching_rejects_feature_manifest_not_used_by_qc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = write_small_matching_inputs(tmp_path)
    feature_manifest_path = paths["radiomics"].with_suffix(".run_manifest.json")
    payload = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    payload["regenerated"] = True
    feature_manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "radiomics_matching.py",
            "--query_csv",
            str(paths["query"]),
            "--reference_csv",
            str(paths["reference"]),
            "--features_csv",
            str(paths["radiomics"]),
            "--out_dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(ValueError, match="did not use this radiomics"):
        RADIOMICS_MATCHING.main()


def test_matching_rejects_upstream_directory_without_overwriting_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = write_small_matching_inputs(tmp_path)
    cohort_manifest_path = tmp_path / "manifest.json"
    before = cohort_manifest_path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "radiomics_matching.py",
            "--query_csv",
            str(paths["query"]),
            "--reference_csv",
            str(paths["reference"]),
            "--features_csv",
            str(paths["radiomics"]),
            "--out_dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(ValueError, match="collides with input location"):
        RADIOMICS_MATCHING.main()
    assert cohort_manifest_path.read_text(encoding="utf-8") == before


def test_matching_clis_write_complete_atomic_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write_small_matching_inputs(tmp_path)
    radiomics_out = tmp_path / "radiomics_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "radiomics_matching.py",
            "--query_csv",
            str(paths["query"]),
            "--reference_csv",
            str(paths["reference"]),
            "--features_csv",
            str(paths["radiomics"]),
            "--out_dir",
            str(radiomics_out),
        ],
    )
    RADIOMICS_MATCHING.main()
    radiomics_manifest = json.loads((radiomics_out / "manifest.json").read_text(encoding="utf-8"))
    assert radiomics_manifest["completed"]
    assert radiomics_manifest["analysis_role"] == "primary"
    assert radiomics_manifest["cohort_policy"] == "primary"
    assert radiomics_manifest["upstream"]["segmentation_manifest"]["sha256"] == "synthetic"
    assert radiomics_manifest["script"]["sha256"] == sha256_file(Path(RADIOMICS_MATCHING.__file__))
    assert all(
        digest == sha256_file(radiomics_out / name)
        for name, digest in radiomics_manifest["outputs"].items()
    )
    topk = pd.read_csv(radiomics_out / "ranking_top4.csv")
    assert {"display_order", "score_midrank", "score_tie_count"} <= set(topk.columns)

    crossfit_out = tmp_path / "crossfit_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crossfit_efa_matching.py",
            "--query_csv",
            str(paths["query"]),
            "--reference_csv",
            str(paths["reference"]),
            "--features_csv",
            str(paths["efa"]),
            "--feature_representation",
            "area_normalized",
            "--out_dir",
            str(crossfit_out),
        ],
    )
    CROSSFIT.main()
    crossfit_manifest = json.loads((crossfit_out / "manifest.json").read_text(encoding="utf-8"))
    assert crossfit_manifest["completed"]
    assert crossfit_manifest["analysis_role"] == "primary"
    assert crossfit_manifest["feature_representation"] == "area_normalized"
    assert crossfit_manifest["n_candidate_configurations"] == 28
    assert crossfit_manifest["upstream"]["segmentation_manifest"]["sha256"] == "synthetic"
    assert all(
        digest == sha256_file(crossfit_out / name)
        for name, digest in crossfit_manifest["outputs"].items()
    )
    audit = pd.read_csv(crossfit_out / "crossfit_selection_audit.csv")
    assert len(audit) == 3 * 28
    assert audit.groupby("held_out_person")["selected"].sum().eq(1).all()


def test_matching_failure_marks_manifest_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write_small_matching_inputs(tmp_path)
    out_dir = tmp_path / "failed_out"
    out_dir.mkdir()
    old_output = out_dir / "true_rank.csv"
    old_output.write_text("previous valid output", encoding="utf-8")
    monkeypatch.setattr(
        RADIOMICS_MATCHING,
        "compute_leave_one_person_out_scores",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "radiomics_matching.py",
            "--query_csv",
            str(paths["query"]),
            "--reference_csv",
            str(paths["reference"]),
            "--features_csv",
            str(paths["radiomics"]),
            "--out_dir",
            str(out_dir),
        ],
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        RADIOMICS_MATCHING.main()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"] is False
    assert old_output.read_text(encoding="utf-8") == "previous valid output"

    crossfit_out = tmp_path / "failed_crossfit"
    crossfit_out.mkdir()
    old_audit = crossfit_out / "crossfit_selection_audit.csv"
    old_audit.write_text("previous valid audit", encoding="utf-8")
    monkeypatch.setattr(
        CROSSFIT,
        "run_crossfit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crossfit_efa_matching.py",
            "--query_csv",
            str(paths["query"]),
            "--reference_csv",
            str(paths["reference"]),
            "--features_csv",
            str(paths["efa"]),
            "--feature_representation",
            "area_normalized",
            "--out_dir",
            str(crossfit_out),
        ],
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        CROSSFIT.main()

    crossfit_manifest = json.loads((crossfit_out / "manifest.json").read_text(encoding="utf-8"))
    assert crossfit_manifest["completed"] is False
    assert old_audit.read_text(encoding="utf-8") == "previous valid audit"
