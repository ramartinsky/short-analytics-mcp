"""Correlation calculations."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy import stats

from .common import Response, finite_floats, json_safe, require_equal_lengths


def calculate_correlation(
    x: list[float], y: list[float], method: Literal["pearson", "spearman"] = "pearson"
) -> Response:
    n = require_equal_lengths(x=x, y=y)
    if n == 0:
        raise ValueError("x and y must not be empty.")
    x_array = finite_floats(x, "x")
    y_array = finite_floats(y, "y")
    warnings = [
        "Association does not establish causation.",
        "The ordinary p-value assumes independent observations.",
    ]
    undefined_reason: str | None = None
    coefficient: float | None = None
    p_value: float | None = None
    confidence_interval: dict[str, float] | None = None

    if n < 2:
        undefined_reason = "At least two paired observations are required."
    elif np.ptp(x_array) == 0 or np.ptp(y_array) == 0:
        undefined_reason = "Correlation is undefined when either array is constant."
    else:
        test = stats.pearsonr(x_array, y_array) if method == "pearson" else stats.spearmanr(x_array, y_array)
        coefficient = float(test.statistic)
        p_value = float(test.pvalue)
        if method == "pearson" and n > 3 and abs(coefficient) < 1:
            z = math.atanh(coefficient)
            delta = stats.norm.ppf(0.975) / math.sqrt(n - 3)
            confidence_interval = {
                "level": 0.95,
                "lower": math.tanh(z - delta),
                "upper": math.tanh(z + delta),
            }

    if coefficient is None:
        direction = "undefined"
    elif abs(coefficient) < 1e-12:
        direction = "no monotonic association" if method == "spearman" else "no linear association"
    else:
        strength = "weak" if abs(coefficient) < 0.3 else "moderate" if abs(coefficient) < 0.7 else "strong"
        direction = f"{strength} {'positive' if coefficient > 0 else 'negative'} " + (
            "monotonic association" if method == "spearman" else "linear association"
        )

    result = {
        "coefficient": coefficient,
        "p_value": p_value,
        "n": n,
        "confidence_interval": confidence_interval,
        "undefined_reason": undefined_reason,
        "explanation": f"{method.title()} correlation indicates {direction}.",
    }
    return Response(result=json_safe(result), warnings=warnings, metadata={"method": method, "rows_used": n, "rows_dropped": 0})

