"""MCP tool registration."""

from __future__ import annotations

import logging
import json
from collections.abc import Callable
from functools import wraps
from typing import Annotated, Any, Literal, ParamSpec, TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent
from pydantic import Field

from . import __version__
from .classification import ValidationScores, calculate_classification_metrics
from .common import Label, Response
from .contingency import calculate_contingency
from .correlation import calculate_correlation
from .prophet_tools import FutureSpec, ProphetSettings, Regressor, fit_prophet_model, forecast_prophet_next

logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")

mcp = MCPServer(
    "statistical-mcp",
    version=__version__,
    instructions="Self-contained, explainable statistical calculations. Outputs describe associations, not causality.",
    log_level="INFO",
)


def tool_errors(function: Callable[P, R]) -> Callable[P, R]:
    """Expose expected validation/calculation messages while logging tracebacks."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Tool call rejected: %s", exc)
            raise ToolError(str(exc)) from exc
        except Exception:
            logger.exception("Unexpected tool failure")
            raise

    return wrapped


def compact(response: Response) -> CallToolResult:
    """Build both MCP result channels from the same validated JSON value."""

    structured = response.model_dump(mode="json")
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(structured, separators=(",", ":"), ensure_ascii=False))],
        structured_content=structured,
    )


@mcp.tool(title="Fit and explain a Prophet model")
@tool_errors
def prophet_model(
    records: Annotated[list[dict[str, str | float | int | None]], Field(min_length=2, description="Inline records; no paths or remote URLs.")],
    timestamp_column: Annotated[str, Field(description="Name of the ISO 8601 timestamp field.")],
    target_column: Annotated[str, Field(description="Name of the numeric target field.")],
    regressors: Annotated[list[Regressor] | None, Field(description="Optional numeric regressors and their interpretation.")] = None,
    settings: ProphetSettings | None = None,
) -> Response:
    """Fit Prophet and summarize trend, seasonal components, changepoints, coefficients, and in-sample error."""

    return compact(fit_prophet_model(records, timestamp_column, target_column, regressors, settings))  # type: ignore[return-value]


@mcp.tool(title="Forecast exactly one next value with Prophet")
@tool_errors
def prophet_next(
    records: Annotated[list[dict[str, str | float | int | None]], Field(min_length=2, description="Inline training records.")],
    timestamp_column: str,
    target_column: str,
    future: Annotated[FutureSpec, Field(description="Exactly one future timestamp or explicit frequency.")],
    future_regressors: Annotated[dict[str, float] | None, Field(description="Required future values for every configured regressor.")] = None,
    regressors: list[Regressor] | None = None,
    settings: ProphetSettings | None = None,
) -> Response:
    """Refit Prophet for this call and predict one future observation, including component contributions."""

    return compact(  # type: ignore[return-value]
        forecast_prophet_next(records, timestamp_column, target_column, future, future_regressors, regressors, settings)
    )


@mcp.tool(title="Correlation and significance")
@tool_errors
def correlation(
    x: Annotated[list[float], Field(min_length=1)],
    y: Annotated[list[float], Field(min_length=1)],
    method: Literal["pearson", "spearman"] = "pearson",
) -> Response:
    """Calculate Pearson or Spearman correlation with assumptions and an optional Pearson interval."""

    return compact(calculate_correlation(x, y, method))  # type: ignore[return-value]


@mcp.tool(title="Categorical association and Simpson analysis")
@tool_errors
def contingency(
    records: Annotated[list[dict[str, Label | None]], Field(min_length=1)],
    row_field: str,
    column_field: str,
    stratum_field: str | None = None,
    exposure_positive: Label | None = None,
    outcome_positive: Label | None = None,
) -> Response:
    """Build a contingency table, select chi-square/Fisher testing, and conservatively detect Simpson reversals."""

    return compact(  # type: ignore[return-value]
        calculate_contingency(records, row_field, column_field, stratum_field, exposure_positive, outcome_positive)
    )


@mcp.tool(title="Binary classification metrics")
@tool_errors
def classification_metrics(
    true_labels: Annotated[list[Label], Field(min_length=1)],
    positive_label: Label,
    predicted_labels: list[Label] | None = None,
    probabilities: list[float] | None = None,
    threshold: Annotated[float, Field(ge=0, le=1)] = 0.5,
    validation: Annotated[ValidationScores | None, Field(description="Separate validation data used only for threshold selection.")] = None,
    threshold_objective: Literal["f1", "youden_j"] = "f1",
) -> Response:
    """Report binary classification metrics and optionally select a threshold only on separate validation data."""

    return compact(  # type: ignore[return-value]
        calculate_classification_metrics(
            true_labels, positive_label, predicted_labels, probabilities, threshold, validation, threshold_objective
        )
    )
