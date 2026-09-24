"""Contingency tables, association tests, and conservative Simpson reversal detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .common import MAX_CATEGORIES, MAX_ROWS, Label, Response, json_safe, label_key


def _same_label(left: Any, right: Label) -> bool:
    return isinstance(left, type(right)) and left == right


def _table_payload(table: pd.DataFrame) -> dict[str, Any]:
    observed = table.to_numpy(dtype=float)
    chi2, p_value, dof, expected = stats.chi2_contingency(observed, correction=False)
    n = observed.sum()
    denominator = min(observed.shape[0] - 1, observed.shape[1] - 1)
    cramers_v = None if denominator <= 0 or n == 0 else float(np.sqrt(chi2 / (n * denominator)))
    sparse = bool((expected < 5).any())
    fisher = None
    selected_test = "chi_square"
    if observed.shape == (2, 2) and sparse:
        odds_ratio, fisher_p = stats.fisher_exact(observed)
        odds_is_finite = bool(np.isfinite(odds_ratio))
        fisher = {
            "odds_ratio": float(odds_ratio) if odds_is_finite else None,
            "odds_ratio_reason": None if odds_is_finite else "The odds ratio is infinite because a cell count is zero.",
            "p_value": float(fisher_p),
            "alternative": "two-sided",
        }
        selected_test = "fisher_exact"
    row_denominators = observed.sum(axis=1, keepdims=True)
    column_denominators = observed.sum(axis=0, keepdims=True)
    row_props = np.divide(observed, row_denominators, out=np.zeros_like(observed), where=row_denominators != 0)
    column_props = np.divide(observed, column_denominators, out=np.zeros_like(observed), where=column_denominators != 0)
    return {
        "row_labels": [str(item) for item in table.index],
        "column_labels": [str(item) for item in table.columns],
        "counts": observed.astype(int).tolist(),
        "row_proportions": row_props.tolist(),
        "column_proportions": column_props.tolist(),
        "expected_counts": expected.tolist(),
        "chi_square": {"statistic": float(chi2), "p_value": float(p_value), "degrees_of_freedom": int(dof)},
        "fisher_exact": fisher,
        "selected_test": selected_test,
        "cramers_v": cramers_v,
        "sparse": sparse,
    }


def _rate_difference(frame: pd.DataFrame, row_field: str, column_field: str, exposure: Label, outcome: Label) -> dict[str, Any] | None:
    exposed = frame[frame[row_field].map(lambda value: _same_label(value, exposure))]
    unexposed = frame[~frame[row_field].map(lambda value: _same_label(value, exposure))]
    if exposed.empty or unexposed.empty:
        return None
    exposed_outcomes = exposed[column_field].map(lambda value: _same_label(value, outcome))
    unexposed_outcomes = unexposed[column_field].map(lambda value: _same_label(value, outcome))
    exposed_rate = float(exposed_outcomes.mean())
    unexposed_rate = float(unexposed_outcomes.mean())
    return {
        "exposed_rate": exposed_rate,
        "unexposed_rate": unexposed_rate,
        "exposed_n": len(exposed),
        "unexposed_n": len(unexposed),
        "difference": exposed_rate - unexposed_rate,
    }


def _simpson_analysis(
    frame: pd.DataFrame,
    row_field: str,
    column_field: str,
    stratum_field: str,
    exposure_positive: Label,
    outcome_positive: Label,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    pooled = _rate_difference(frame, row_field, column_field, exposure_positive, outcome_positive)
    if pooled is None:
        raise ValueError("Both exposure groups are required in the pooled data.")
    strata: list[dict[str, Any]] = []
    differences: list[float] = []
    missing_groups: list[str] = []
    sparse_strata: list[str] = []
    for stratum, subset in frame.groupby(stratum_field, dropna=False, sort=True):
        rates = _rate_difference(subset, row_field, column_field, exposure_positive, outcome_positive)
        name = str(stratum)
        if rates is None:
            missing_groups.append(name)
            strata.append({"stratum": name, "evaluable": False, "reason": "One exposure group is missing."})
            continue
        if min(int(rates["exposed_n"]), int(rates["unexposed_n"])) < 5:
            sparse_strata.append(name)
        difference = float(rates["difference"])
        if difference != 0:
            differences.append(difference)
        strata.append({"stratum": name, "evaluable": True, **rates})

    directions = {1 if item > 0 else -1 for item in differences}
    pooled_direction = 1 if pooled["difference"] > 0 else -1 if pooled["difference"] < 0 else 0
    reversal = bool(differences and len(directions) == 1 and pooled_direction != 0 and pooled_direction not in directions)
    mixed = len(directions) > 1
    if mixed:
        warnings.append("Stratum-specific differences have mixed directions; no Simpson reversal is flagged.")
    if missing_groups:
        warnings.append("Some strata are not evaluable because an exposure group is missing.")
    if sparse_strata:
        warnings.append("Some strata are sparse; rate differences may be unstable.")
    explanation = (
        "An observed Simpson reversal is present: every evaluable nonzero stratum difference points opposite to the pooled difference. "
        "Changing exposure-group composition across strata can produce this pattern; it is not proof of confounding or causality."
        if reversal
        else "No Simpson reversal meeting the strict same-direction rule was observed. Group composition can still affect the pooled estimate."
    )
    return {
        "evaluated": True,
        "reversal": reversal,
        "mixed_stratum_directions": mixed,
        "pooled": pooled,
        "strata": strata,
        "missing_group_strata": missing_groups,
        "sparse_strata": sparse_strata,
        "direction": f"outcome rate for exposure={exposure_positive!r} minus the other exposure group",
        "explanation": explanation,
    }, warnings


def calculate_contingency(
    records: list[dict[str, Label | None]],
    row_field: str,
    column_field: str,
    stratum_field: str | None = None,
    exposure_positive: Label | None = None,
    outcome_positive: Label | None = None,
) -> Response:
    if not records:
        raise ValueError("records must not be empty.")
    if len(records) > MAX_ROWS:
        raise ValueError(f"records exceeds the {MAX_ROWS} row limit.")
    required = {row_field, column_field} | ({stratum_field} if stratum_field else set())
    missing = sorted({field for field in required if any(field not in record for record in records)})
    if missing:
        raise ValueError(f"Every record must contain: {', '.join(missing)}.")
    frame = pd.DataFrame(records)
    before = len(frame)
    frame = frame.dropna(subset=list(required))
    if frame.empty:
        raise ValueError("No complete categorical records remain after dropping nulls.")
    for field in required:
        if frame[field].nunique() > MAX_CATEGORIES:
            raise ValueError(f"{field} exceeds the {MAX_CATEGORIES} category limit.")
    table = pd.crosstab(frame[row_field], frame[column_field], dropna=False)
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("A contingency table requires at least two row and two column categories.")
    warnings: list[str] = []
    payload = _table_payload(table)
    if payload["sparse"] and table.shape != (2, 2):
        warnings.append("Expected counts below 5 make the larger-table chi-square approximation potentially unreliable.")
    elif payload["selected_test"] == "fisher_exact":
        warnings.append("Sparse 2x2 table: Fisher's exact test is selected over the asymptotic chi-square test.")

    simpson: dict[str, Any] = {"evaluated": False, "reason": "A stratum field and explicit binary labels are required."}
    if stratum_field is not None:
        binary = table.shape == (2, 2)
        if binary and exposure_positive is not None and outcome_positive is not None:
            row_keys = {label_key(item) for item in frame[row_field]}
            column_keys = {label_key(item) for item in frame[column_field]}
            if label_key(exposure_positive) not in row_keys or label_key(outcome_positive) not in column_keys:
                raise ValueError("The explicit exposure/outcome labels must occur in their respective fields.")
            simpson, simpson_warnings = _simpson_analysis(
                frame, row_field, column_field, stratum_field, exposure_positive, outcome_positive
            )
            warnings.extend(simpson_warnings)
        elif binary:
            warnings.append("Binary dimensions detected, but Simpson analysis needs exposure_positive and outcome_positive to fix direction.")
        else:
            simpson = {"evaluated": False, "reason": "Automatic Simpson detection applies only to binary row and column dimensions."}

    payload["simpson"] = simpson
    return Response(
        result=json_safe(payload),
        warnings=warnings,
        metadata={"rows_used": len(frame), "rows_dropped": before - len(frame), "method": payload["selected_test"]},
    )
