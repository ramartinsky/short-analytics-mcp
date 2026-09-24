import pytest

from statistical_mcp.correlation import calculate_correlation


def test_known_pearson_correlation() -> None:
    response = calculate_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert response.result["coefficient"] == pytest.approx(1.0)
    assert response.result["p_value"] == pytest.approx(0.0)
    assert response.result["n"] == 5


def test_constant_array_is_valid_but_undefined() -> None:
    response = calculate_correlation([1, 1, 1], [1, 2, 3])
    assert response.result["coefficient"] is None
    assert "constant" in response.result["undefined_reason"]


def test_mismatched_lengths_are_invalid() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        calculate_correlation([1, 2], [1])

