import pytest

from statistical_mcp.contingency import calculate_contingency


def _records(stratum: str, exposed_yes: int, exposed_no: int, control_yes: int, control_no: int):
    return (
        [{"group": "treated", "outcome": "yes", "stratum": stratum}] * exposed_yes
        + [{"group": "treated", "outcome": "no", "stratum": stratum}] * exposed_no
        + [{"group": "control", "outcome": "yes", "stratum": stratum}] * control_yes
        + [{"group": "control", "outcome": "no", "stratum": stratum}] * control_no
    )


def test_known_contingency_counts_and_fisher_selection() -> None:
    records = (
        [{"row": "a", "col": "yes"}] * 1
        + [{"row": "a", "col": "no"}] * 9
        + [{"row": "b", "col": "yes"}] * 8
        + [{"row": "b", "col": "no"}] * 2
    )
    response = calculate_contingency(records, "row", "col")
    assert response.result["selected_test"] == "fisher_exact"
    assert response.result["fisher_exact"]["p_value"] < 0.01


def test_detects_strict_simpson_reversal() -> None:
    records = _records("easy", 9, 1, 80, 20) + _records("hard", 20, 80, 1, 9)
    response = calculate_contingency(records, "group", "outcome", "stratum", "treated", "yes")
    simpson = response.result["simpson"]
    assert simpson["reversal"] is True
    assert simpson["pooled"]["difference"] < 0
    assert all(item["difference"] > 0 for item in simpson["strata"] if item["evaluable"])


def test_does_not_flag_mixed_stratum_directions() -> None:
    records = _records("one", 8, 2, 6, 4) + _records("two", 2, 8, 4, 6)
    response = calculate_contingency(records, "group", "outcome", "stratum", "treated", "yes")
    assert response.result["simpson"]["reversal"] is False
    assert response.result["simpson"]["mixed_stratum_directions"] is True

