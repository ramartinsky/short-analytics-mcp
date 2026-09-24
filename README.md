# Statistical MCP server

A stateless Python MCP server for five explainable statistical operations: Prophet model inspection, one-step Prophet forecasts, correlation, contingency tables (including conservative Simpson reversal detection), and binary classification metrics.

Every call is self-contained. Inputs are inline JSON, outputs contain `result`, `warnings`, and `metadata`, and undefined statistics are JSON `null` with a reason. The server does not fetch remote data, read user paths, persist models, or require external services.

## Run with Docker

The supported image target is `linux/amd64`.

```bash
docker build --platform linux/amd64 -t statistical-mcp .
docker run --rm -p 8000:8000 statistical-mcp
```

Streamable HTTP is available at `http://localhost:8000/mcp`. The HTTP server is stateless and uses one JSON response per request. To run stdio for a desktop client:

```bash
docker run --rm -i statistical-mcp --transport stdio
```

A desktop MCP configuration can launch the local image like this:

```json
{
  "mcpServers": {
    "statistics": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "statistical-mcp", "--transport", "stdio"]
    }
  }
}
```

For local development, install the locked environment and run either transport:

```bash
uv sync --frozen --extra dev
uv run statistical-mcp --transport streamable-http --host 127.0.0.1 --port 8000
uv run statistical-mcp --transport stdio
uv run pytest
```

Only `--transport`, `--host`, `--port`, and `--max-request-bytes` are configurable. Logs go to stderr so stdio stdout remains protocol-only. Put an authenticated reverse proxy in front of remote HTTP deployments.

## Tool examples

These are the `arguments` objects passed to MCP `tools/call`.

### `correlation`

```json
{"x":[1,2,3,4],"y":[2,3,5,8],"method":"pearson"}
```

Returns the coefficient, p-value, paired sample count, explanation, and a 95% Pearson interval when it is defined. Constant arrays are valid data but produce `null` statistics. P-values assume independent observations; association is not causation.

### `contingency`

```json
{
  "records": [
    {"treatment":"yes","outcome":"recovered","clinic":"A"},
    {"treatment":"no","outcome":"not_recovered","clinic":"A"},
    {"treatment":"yes","outcome":"recovered","clinic":"B"},
    {"treatment":"no","outcome":"recovered","clinic":"B"}
  ],
  "row_field":"treatment",
  "column_field":"outcome",
  "stratum_field":"clinic",
  "exposure_positive":"yes",
  "outcome_positive":"recovered"
}
```

Returns counts, row/column proportions, expected counts, chi-square results, Cramer's V, and Fisher's exact test for sparse 2×2 tables. Simpson analysis fixes its direction as the positive-exposure outcome rate minus the other exposure group's rate. A reversal is flagged only when every evaluable nonzero stratum difference has one direction and the pooled difference has the opposite direction. It describes an observed pattern, not causality.

### `classification_metrics`

```json
{
  "true_labels":[0,0,1,1],
  "positive_label":1,
  "predicted_labels":[0,1,1,1],
  "probabilities":[0.1,0.6,0.7,0.9],
  "validation":{"labels":[0,0,1,1],"probabilities":[0.2,0.4,0.65,0.8]},
  "threshold_objective":"f1"
}
```

Predicted labels determine precision, recall, accuracy, F1, and the confusion matrix when supplied; probabilities determine ROC AUC. A threshold is suggested only from the separate `validation` set, with highest-threshold tie breaking, then evaluated separately on the evaluation scores. The threshold depends on the chosen F1 or Youden's J objective.

### `prophet_model`

```json
{
  "records":[
    {"date":"2026-01-01","sales":10,"promotion":0},
    {"date":"2026-01-02","sales":12,"promotion":1},
    {"date":"2026-01-03","sales":11,"promotion":0}
  ],
  "timestamp_column":"date",
  "target_column":"sales",
  "regressors":[{"name":"promotion","mode":"additive","unit":"flag"}],
  "settings":{"weekly_seasonality":false,"daily_seasonality":false}
}
```

Returns component summaries, changepoints, extra-regressor coefficients on the original data scale, and in-sample MAE. Additive coefficients are target change per regressor unit; multiplicative coefficients are relative to trend. Seasonal Fourier weights are never presented as ordinary coefficients. Default MAP fitting does not report coefficient intervals; bounded MCMC (`mcmc_samples` up to 300) produces posterior intervals. Fitted effects are associations, not causal effects.

### `prophet_next`

```json
{
  "records":[
    {"date":"2026-01-01","sales":10,"promotion":0},
    {"date":"2026-01-02","sales":12,"promotion":1},
    {"date":"2026-01-03","sales":11,"promotion":0}
  ],
  "timestamp_column":"date",
  "target_column":"sales",
  "future":{"timestamp":"2026-01-04"},
  "regressors":[{"name":"promotion"}],
  "future_regressors":{"promotion":1},
  "settings":{"weekly_seasonality":false,"daily_seasonality":false}
}
```

The model is refit inside the request and exactly one future row is predicted. Instead of a timestamp, `future` may contain an explicit pandas-compatible `frequency`, such as `{"frequency":"1D"}`. `{"frequency":"infer"}` is accepted only when every training interval is identical. Every configured regressor needs a future value. Duplicate timestamps, mixed timezone conventions, and a timestamp at or before the final observation are rejected; timezone-aware data is normalized to UTC.

## Limits and assumptions

- At most 10,000 rows per call and 200 categories per categorical field.
- HTTP request bodies default to 4 MiB; `--max-request-bytes` can lower or raise this deployment limit.
- Prophet fitting has two in-process slots. Calls beyond that capacity receive a retryable tool error instead of queuing without bound.
- Prophet accepts linear or flat growth and a small allowlist of trend/seasonality settings. No runtime downloads are performed.
- Correlation and contingency p-values are ordinary unadjusted tests. Larger sparse contingency tables are explicitly flagged as unreliable for asymptotic inference.
- Classification is binary. ROC AUC requires probabilities and both true classes. Undefined precision, recall, F1, and AUC are `null` with reasons.
- JSON never contains NaN or Infinity.

Input and output JSON Schemas are published by MCP tool discovery. Tool results include structured content and the same compact JSON in a text block for text-only clients.
