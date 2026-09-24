"""Shared Prophet validation, fitting, summaries, and one-step forecasting."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

# Prophet imports its optional plotting module eagerly and logs an error when
# Plotly is absent even though this server never plots. Keep dependency chatter
# out of protocol-facing processes and retain only our own concise fit errors.
logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from pydantic import Field, model_validator
from typing_extensions import Annotated

from .common import MAX_MCMC_SAMPLES, MAX_ROWS, Response, StrictModel, json_safe

_FIT_SLOTS = threading.BoundedSemaphore(2)
_cmdstan_logger = logging.getLogger("cmdstanpy")
_cmdstan_logger.addHandler(logging.NullHandler())
_cmdstan_logger.setLevel(logging.WARNING)


class Regressor(StrictModel):
    name: str = Field(min_length=1)
    mode: Literal["additive", "multiplicative"] = "additive"
    unit: str | None = None
    standardize: bool | Literal["auto"] = "auto"


SeasonalitySetting = bool | Literal["auto"] | Annotated[int, Field(gt=0, le=50)]


class ProphetSettings(StrictModel):
    growth: Literal["linear", "flat"] = "linear"
    seasonality_mode: Literal["additive", "multiplicative"] = "additive"
    yearly_seasonality: SeasonalitySetting = "auto"
    weekly_seasonality: SeasonalitySetting = "auto"
    daily_seasonality: SeasonalitySetting = "auto"
    changepoint_prior_scale: float = Field(default=0.05, gt=0, le=1)
    seasonality_prior_scale: float = Field(default=10.0, gt=0, le=100)
    interval_width: float = Field(default=0.8, gt=0, lt=1)
    n_changepoints: int = Field(default=25, ge=0, le=100)
    mcmc_samples: int = Field(default=0, ge=0, le=MAX_MCMC_SAMPLES)


class FutureSpec(StrictModel):
    timestamp: str | None = None
    frequency: str | None = None

    @model_validator(mode="after")
    def exactly_one(self) -> "FutureSpec":
        if (self.timestamp is None) == (self.frequency is None):
            raise ValueError("Supply exactly one of timestamp or frequency.")
        return self


@contextmanager
def _fit_slot():
    if not _FIT_SLOTS.acquire(blocking=False):
        raise RuntimeError("Forecast fitting capacity is busy; retry after another fit completes.")
    try:
        yield
    finally:
        _FIT_SLOTS.release()


def _timestamp(value: Any) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ValueError(f"Timestamp must be an ISO 8601 string, got {type(value).__name__}.")
    try:
        return pd.Timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except Exception as exc:
        raise ValueError(f"Invalid ISO 8601 timestamp: {value!r}.") from exc


def _prepare(
    records: list[dict[str, str | float | int | None]],
    timestamp_column: str,
    target_column: str,
    regressors: list[Regressor],
) -> tuple[pd.DataFrame, list[str], dict[str, Any], bool]:
    if not records:
        raise ValueError("records must not be empty.")
    if len(records) > MAX_ROWS:
        raise ValueError(f"records exceeds the {MAX_ROWS} row limit.")
    names = [item.name for item in regressors]
    if len(names) != len(set(names)):
        raise ValueError("Regressor names must be unique.")
    required = [timestamp_column, target_column, *names]
    absent = [name for name in required if any(name not in record for record in records)]
    if absent:
        raise ValueError(f"Every record must contain: {', '.join(sorted(set(absent)))}.")
    frame = pd.DataFrame(records)[required].copy()
    aware_flags: list[bool] = []
    parsed: list[pd.Timestamp] = []
    for raw in frame[timestamp_column]:
        stamp = _timestamp(raw)
        aware = stamp.tzinfo is not None and stamp.utcoffset() is not None
        aware_flags.append(aware)
        parsed.append(stamp)
    if len(set(aware_flags)) > 1:
        raise ValueError("Mixed timezone-aware and timezone-naive timestamps are not allowed.")
    timezone_aware = bool(aware_flags[0])
    if timezone_aware:
        dates = pd.DatetimeIndex([stamp.tz_convert("UTC").tz_localize(None) for stamp in parsed])
    else:
        dates = pd.DatetimeIndex(parsed)
    if dates.duplicated().any():
        raise ValueError("Duplicate timestamps are not allowed.")
    frame["ds"] = dates
    frame["y"] = pd.to_numeric(frame[target_column], errors="coerce")
    for name in names:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    before = len(frame)
    frame = frame.dropna(subset=["y", *names]).sort_values("ds").reset_index(drop=True)
    numeric = frame[["y", *names]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Target and regressor values must be finite numbers.")
    if len(frame) < 2:
        raise ValueError("At least two complete observations are required.")
    warnings: list[str] = []
    if len(frame) < 30:
        warnings.append("Short history: trend, seasonality, and interval estimates may be unstable.")
    constant = [name for name in names if frame[name].nunique() <= 1]
    if constant:
        warnings.append(f"Constant regressors detected: {', '.join(constant)}.")
    if len(names) > 1 and np.linalg.matrix_rank(frame[names].to_numpy(dtype=float)) < len(names):
        warnings.append("Regressors are collinear; individual coefficients may be unstable.")
    metadata = {"rows_used": len(frame), "rows_dropped": before - len(frame), "timezone": "UTC" if timezone_aware else "naive"}
    return frame[["ds", "y", *names]], warnings, metadata, timezone_aware


def _make_model(settings: ProphetSettings, regressors: list[Regressor]) -> Prophet:
    model = Prophet(**settings.model_dump())
    for item in regressors:
        model.add_regressor(item.name, mode=item.mode, standardize=item.standardize)
    return model


def _fit(frame: pd.DataFrame, settings: ProphetSettings, regressors: list[Regressor]) -> Prophet:
    with _fit_slot():
        return _make_model(settings, regressors).fit(frame)


def _iso(stamp: pd.Timestamp, aware: bool) -> str:
    value = pd.Timestamp(stamp)
    if aware:
        return value.tz_localize("UTC").isoformat().replace("+00:00", "Z")
    return value.isoformat()


def _component_summaries(forecast: pd.DataFrame) -> list[dict[str, Any]]:
    candidates = ["trend", "yearly", "weekly", "daily", "additive_terms", "multiplicative_terms"]
    summaries = []
    for name in candidates:
        if name in forecast:
            values = forecast[name].to_numpy(dtype=float)
            summaries.append({"name": name, "mean": float(values.mean()), "minimum": float(values.min()), "maximum": float(values.max()), "last": float(values[-1])})
    return summaries


def fit_prophet_model(
    records: list[dict[str, str | float | int | None]],
    timestamp_column: str,
    target_column: str,
    regressors: list[Regressor] | None = None,
    settings: ProphetSettings | None = None,
) -> Response:
    regressors = regressors or []
    settings = settings or ProphetSettings()
    frame, warnings, metadata, aware = _prepare(records, timestamp_column, target_column, regressors)
    model = _fit(frame, settings, regressors)
    forecast = model.predict(frame.drop(columns="y"))
    coefficients: list[dict[str, Any]] = []
    if regressors:
        by_name = {item.name: item for item in regressors}
        for row in regressor_coefficients(model).to_dict(orient="records"):
            config = by_name[str(row["regressor"])]
            coefficient = {
                "name": row["regressor"],
                "mode": row["regressor_mode"],
                "unit": config.unit,
                "coefficient": row["coef"],
                "explanation": (
                    "Estimated target change per one-unit regressor increase."
                    if row["regressor_mode"] == "additive"
                    else "Estimated effect per unit relative to the trend."
                ),
            }
            if settings.mcmc_samples > 0:
                coefficient["posterior_interval"] = {"level": settings.interval_width, "lower": row["coef_lower"], "upper": row["coef_upper"]}
            coefficients.append(coefficient)
    warnings.append("All effects are fitted associations, not causal effects; diagnostics are in-sample.")
    if settings.mcmc_samples == 0:
        warnings.append("MAP fitting was used; coefficient confidence bounds are not reported.")
    result = {
        "components": _component_summaries(forecast),
        "component_explanations": {
            "trend": "Fitted baseline trajectory.",
            "seasonal": "Periodic Fourier components; these are not ordinary feature coefficients.",
        },
        "changepoints": [_iso(item, aware) for item in model.changepoints],
        "coefficients": coefficients,
        "in_sample": {"mae": float(np.mean(np.abs(frame["y"].to_numpy() - forecast["yhat"].to_numpy())))},
    }
    metadata.update({"method": "prophet", "fit": "mcmc" if settings.mcmc_samples else "map", "settings": settings.model_dump()})
    return Response(result=json_safe(result), warnings=warnings, metadata=json_safe(metadata))


def forecast_prophet_next(
    records: list[dict[str, str | float | int | None]],
    timestamp_column: str,
    target_column: str,
    future: FutureSpec,
    future_regressors: dict[str, float] | None = None,
    regressors: list[Regressor] | None = None,
    settings: ProphetSettings | None = None,
) -> Response:
    regressors = regressors or []
    settings = settings or ProphetSettings()
    frame, warnings, metadata, aware = _prepare(records, timestamp_column, target_column, regressors)
    last = pd.Timestamp(frame["ds"].iloc[-1])
    if future.timestamp is not None:
        raw_future = _timestamp(future.timestamp)
        future_aware = raw_future.tzinfo is not None and raw_future.utcoffset() is not None
        if future_aware != aware:
            raise ValueError("The future timestamp must use the same timezone convention as training timestamps.")
        next_stamp = raw_future.tz_convert("UTC").tz_localize(None) if aware else raw_future
    else:
        assert future.frequency is not None
        if future.frequency == "infer":
            differences = frame["ds"].diff().dropna()
            if differences.empty or differences.nunique() != 1:
                raise ValueError("Frequency can be inferred only from a regular training series.")
            offset = differences.iloc[0]
        else:
            try:
                offset = pd.tseries.frequencies.to_offset(future.frequency)
            except ValueError as exc:
                raise ValueError(f"Invalid explicit frequency: {future.frequency!r}.") from exc
        next_stamp = last + offset
    if next_stamp <= last:
        raise ValueError("The future timestamp must be after the final training observation.")
    supplied = future_regressors or {}
    expected_names = {item.name for item in regressors}
    if set(supplied) != expected_names:
        missing = sorted(expected_names - set(supplied))
        extra = sorted(set(supplied) - expected_names)
        details = [*(f"missing {item}" for item in missing), *(f"unexpected {item}" for item in extra)]
        raise ValueError("future_regressors must match configured regressors: " + ", ".join(details))
    if not all(np.isfinite(float(value)) for value in supplied.values()):
        raise ValueError("Future regressor values must be finite numbers.")
    model = _fit(frame, settings, regressors)
    future_frame = pd.DataFrame([{"ds": next_stamp, **supplied}])
    row = model.predict(future_frame).iloc[0]
    seasonal_names = [name for name in ("yearly", "weekly", "daily", "additive_terms", "multiplicative_terms") if name in row]
    contributions = {"trend": row["trend"], **{name: row[name] for name in seasonal_names}}
    for item in regressors:
        if item.name in row:
            contributions[f"regressor:{item.name}"] = row[item.name]
    result = {
        "timestamp": _iso(next_stamp, aware),
        "yhat": row["yhat"],
        "lower": row["yhat_lower"],
        "upper": row["yhat_upper"],
        "interval_level": settings.interval_width,
        "contributions": contributions,
    }
    metadata.update({"method": "prophet", "fit": "mcmc" if settings.mcmc_samples else "map", "settings": settings.model_dump()})
    return Response(result=json_safe(result), warnings=warnings, metadata=json_safe(metadata))
