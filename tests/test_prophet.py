import math

import pytest

from statistical_mcp.prophet_tools import FutureSpec, Regressor, fit_prophet_model, forecast_prophet_next


def _series():
    return [
        {"when": f"2026-01-{day:02d}", "value": 10 + day * 0.5 + math.sin(day), "promo": day % 2}
        for day in range(1, 29)
    ]


@pytest.mark.prophet
def test_synthetic_fit_and_next_forecast() -> None:
    regressors = [Regressor(name="promo", unit="flag")]
    fitted = fit_prophet_model(_series(), "when", "value", regressors)
    assert fitted.metadata["fit"] == "map"
    assert fitted.result["coefficients"][0]["name"] == "promo"
    assert "posterior_interval" not in fitted.result["coefficients"][0]

    forecast = forecast_prophet_next(
        _series(), "when", "value", FutureSpec(timestamp="2026-01-29"), {"promo": 1}, regressors
    )
    assert forecast.result["timestamp"] == "2026-01-29T00:00:00"
    assert forecast.result["lower"] <= forecast.result["yhat"] <= forecast.result["upper"]


@pytest.mark.prophet
def test_rejects_missing_future_regressor() -> None:
    with pytest.raises(ValueError, match="future_regressors"):
        forecast_prophet_next(
            _series(), "when", "value", FutureSpec(timestamp="2026-01-29"), {}, [Regressor(name="promo")]
        )


@pytest.mark.prophet
def test_frequency_inference_requires_regular_series() -> None:
    regular = forecast_prophet_next(_series(), "when", "value", FutureSpec(frequency="infer"), {"promo": 1}, [Regressor(name="promo")])
    assert regular.result["timestamp"] == "2026-01-29T00:00:00"
    irregular = _series()
    irregular.pop(10)
    with pytest.raises(ValueError, match="regular training series"):
        forecast_prophet_next(irregular, "when", "value", FutureSpec(frequency="infer"), {"promo": 1}, [Regressor(name="promo")])
