import pytest

from statistical_mcp.classification import ValidationScores, calculate_classification_metrics


def test_labels_drive_metrics_and_scores_drive_auc() -> None:
    response = calculate_classification_metrics(
        [0, 0, 1, 1], 1, predicted_labels=[0, 0, 0, 0], probabilities=[0.1, 0.2, 0.8, 0.9]
    )
    assert response.result["recall"] == 0
    assert response.result["roc_auc"] == pytest.approx(1.0)
    assert response.metadata["classification_source"] == "predicted_labels"


def test_auc_and_precision_have_explicit_undefined_reasons() -> None:
    response = calculate_classification_metrics([1, 1], 1, probabilities=[0.1, 0.2], threshold=0.5)
    assert response.result["precision"] is None
    assert response.result["roc_auc"] is None
    assert "precision" in response.result["undefined_reasons"]
    assert "roc_auc" in response.result["undefined_reasons"]


def test_threshold_is_selected_on_separate_validation_data() -> None:
    validation = ValidationScores(labels=[0, 0, 1, 1], probabilities=[0.1, 0.4, 0.6, 0.9])
    response = calculate_classification_metrics(
        [0, 1], 1, probabilities=[0.55, 0.56], validation=validation, threshold_objective="f1"
    )
    assert response.result["threshold_selection"]["threshold"] == pytest.approx(0.6)
    assert response.result["evaluation_at_selected_threshold"]["accuracy"] == pytest.approx(0.5)


def test_rejects_multiclass_truth() -> None:
    with pytest.raises(ValueError, match="at most two classes"):
        calculate_classification_metrics([0, 1, 2], 1, predicted_labels=[0, 1, 2])

