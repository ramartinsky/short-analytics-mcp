"""Binary classification metrics and validation-only threshold selection."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from .common import Label, Response, StrictModel, finite_floats, json_safe, label_key, require_equal_lengths


class ValidationScores(StrictModel):
    labels: list[Label] = Field(min_length=1)
    probabilities: list[float] = Field(min_length=1)


def _binary(labels: list[Label], positive_label: Label, name: str) -> np.ndarray:
    positive_key = label_key(positive_label)
    unique = {label_key(item) for item in labels}
    if positive_key not in unique:
        raise ValueError(f"{name} does not contain the positive_label.")
    if len(unique) > 2:
        raise ValueError(f"{name} must contain at most two classes for binary classification.")
    return np.asarray([label_key(item) == positive_key for item in labels], dtype=bool)


def _classification_values(truth: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[False, True]).ravel()
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": float(accuracy_score(truth, predicted)),
        "f1": f1,
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
        "support": {"negative": int((~truth).sum()), "positive": int(truth.sum())},
        "undefined_reasons": {
            **({"precision": "No positive predictions."} if precision is None else {}),
            **({"recall": "No positive examples."} if recall is None else {}),
            **({"f1": "Precision or recall is undefined, or both are zero."} if f1 is None else {}),
        },
    }


def _select_threshold(truth: np.ndarray, scores: np.ndarray, objective: str) -> tuple[float, dict[str, object]]:
    best_value = -float("inf")
    best_threshold = 0.5
    best_metrics: dict[str, object] = {}
    for threshold in sorted(set(float(item) for item in scores), reverse=True):
        values = _classification_values(truth, scores >= threshold)
        if objective == "f1":
            objective_value = values["f1"]
            score = -float("inf") if objective_value is None else float(objective_value)
        else:
            matrix = values["confusion_matrix"]
            assert isinstance(matrix, dict)
            tp, fn = int(matrix["true_positive"]), int(matrix["false_negative"])
            tn, fp = int(matrix["true_negative"]), int(matrix["false_positive"])
            score = (tp / (tp + fn) if tp + fn else 0) + (tn / (tn + fp) if tn + fp else 0) - 1
        if score > best_value:
            best_value, best_threshold, best_metrics = score, threshold, values
    return best_threshold, best_metrics


def calculate_classification_metrics(
    true_labels: list[Label],
    positive_label: Label,
    predicted_labels: list[Label] | None = None,
    probabilities: list[float] | None = None,
    threshold: float = 0.5,
    validation: ValidationScores | None = None,
    threshold_objective: Literal["f1", "youden_j"] = "f1",
) -> Response:
    if predicted_labels is None and probabilities is None:
        raise ValueError("Supply predicted_labels and/or probabilities.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    n = len(true_labels)
    if n == 0:
        raise ValueError("true_labels must not be empty.")
    truth = _binary(true_labels, positive_label, "true_labels")
    scores = None
    if probabilities is not None:
        require_equal_lengths(true_labels=true_labels, probabilities=probabilities)
        scores = finite_floats(probabilities, "probabilities")
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError("probabilities must be between 0 and 1.")

    selected: dict[str, object] | None = None
    evaluation_threshold = threshold
    if validation is not None:
        require_equal_lengths(validation_labels=validation.labels, validation_probabilities=validation.probabilities)
        validation_truth = _binary(validation.labels, positive_label, "validation.labels")
        validation_scores = finite_floats(validation.probabilities, "validation.probabilities")
        if ((validation_scores < 0) | (validation_scores > 1)).any():
            raise ValueError("validation probabilities must be between 0 and 1.")
        evaluation_threshold, validation_metrics = _select_threshold(validation_truth, validation_scores, threshold_objective)
        selected = {"objective": threshold_objective, "threshold": evaluation_threshold, "validation_metrics": validation_metrics}

    if predicted_labels is not None:
        require_equal_lengths(true_labels=true_labels, predicted_labels=predicted_labels)
        predicted_classes = {label_key(item) for item in predicted_labels}
        if len(predicted_classes) > 2:
            raise ValueError("predicted_labels must contain at most two classes for binary classification.")
        predictions = np.asarray([label_key(item) == label_key(positive_label) for item in predicted_labels], dtype=bool)
        prediction_source = "predicted_labels"
    else:
        assert scores is not None
        predictions = scores >= evaluation_threshold
        prediction_source = "probabilities"

    values = _classification_values(truth, predictions)
    auc_reason = None
    auc = None
    if scores is None:
        auc_reason = "Positive-class probabilities were not supplied."
    elif len(np.unique(truth)) < 2:
        auc_reason = "ROC AUC requires both classes in true_labels."
    else:
        auc = float(roc_auc_score(truth, scores))
    values["roc_auc"] = auc
    if auc_reason:
        values["undefined_reasons"] = {**values["undefined_reasons"], "roc_auc": auc_reason}
    values["threshold_selection"] = selected
    if scores is not None:
        values["evaluation_at_selected_threshold"] = _classification_values(truth, scores >= evaluation_threshold) if selected else None

    positive_rate = float(truth.mean())
    warnings = ["A suggested threshold depends on the chosen validation objective."] if selected else []
    if positive_rate < 0.2 or positive_rate > 0.8:
        warnings.append(f"Class imbalance detected: positive prevalence is {positive_rate:.1%}.")
    return Response(
        result=json_safe(values),
        warnings=warnings,
        metadata={"rows_used": n, "rows_dropped": 0, "classification_source": prediction_source, "threshold": evaluation_threshold},
    )
