# Statistical MCP server — implementation plan

## Goal

One Docker image, defined by a Dockerfile, running a small Python MCP server. Typed JSON inputs, compact explainable outputs, and no database, model registry, UI, scheduler, or background management. Each call is self-contained.

## Runtime and compatibility

- Use the official MCP Python SDK; pin a tested stable release and all runtime dependencies in a lockfile.
- Support Streamable HTTP at `/mcp` (default, port 8000) and stdio through `--transport stdio` for desktop clients. Use stateless HTTP with JSON responses where supported by the pinned SDK.
- Publish input/output JSON schemas. Return MCP structured content plus an equivalent compact JSON text block for clients that consume text only.
- Write logs only to stderr; stdout is reserved for the stdio protocol. No progress chatter or raw library output.
- Use Python 3.12 on a slim Debian base, a non-root user, exec-form entrypoint, and build-time installation of Prophet and its runtime assets. No downloads at startup.
- Start with tested `linux/amd64` support; add `linux/arm64` only after Prophet installation and forecast smoke tests pass there.
- Keep configuration to transport, host, port, and documented request-size/row limits. No mandatory external services; remote deployments can use an existing authenticated reverse proxy.

## Shared contract

- Accept inline JSON records or typed arrays; use ISO 8601 timestamps and explicit column names. No remote data fetching or filesystem paths.
- Return `result`, `warnings`, and compact `metadata` (rows used/dropped, method, relevant settings).
- Reject invalid types, mismatched lengths.
- Represent undefined statistics as `null` with a reason, never NaN/Infinity. Distinguish invalid input from valid but statistically undefined results.
- Use MCP tool errors for calculation/validation failures; keep tracebacks in stderr logs. Bound fitting concurrency and expose clear overload errors.

## Tools

### 1. `prophet_model`

- Input: timestamp/target records, optional numeric regressors, and a small allowlist of trend/seasonality settings.
- Fit Prophet and return trend and seasonal component summaries, changepoints, and extra-regressor coefficients on the original data scale.
- Explain additive coefficients as target change per unit of regressor; explain multiplicative coefficients as effects relative to the trend. Return regressor names, units when supplied, and modes.
- Do not present seasonal Fourier weights as ordinary feature coefficients. Without extra regressors, return an empty coefficient list and component explanations.
- Default to fast MAP fitting. Do not label MAP coefficient bounds as meaningful confidence intervals; posterior coefficient intervals require explicit, bounded MCMC sampling.
- Flag constant/collinear regressors and short histories. Explanations describe fitted associations, not causal effects. Label any fit diagnostics as in-sample.

### 2. `prophet_next`

- Input: the same training data/settings, plus one future timestamp or an explicit frequency. Infer frequency only for a regular series.
- Refit within the request and predict exactly one next value; require future values for every supplied regressor.
- Return timestamp, `yhat`, lower/upper prediction bounds, interval level, and trend/seasonal/regressor contributions.
- Reject duplicate timestamps and future timestamps at or before the last observation. Normalize timezone-aware timestamps to UTC before fitting; reject mixed timezone conventions.
- Share fitting/validation code with `prophet_model`; use no persistent model handles or implicit cached state.

### 3. `correlation`

- Input: two numeric arrays; method `pearson` (default) or `spearman`.
- Return coefficient, p-value, sample count, and a short direction/method explanation; provide a Pearson confidence interval when defined.
- Handle constant arrays and insufficient samples explicitly. State that association is not causation and ordinary p-values assume independent observations.

### 4. `contingency`

- Input: categorical records with row/column fields and an optional stratum field.
- Return counts, row/column proportions, expected counts, chi-square test, and Cramer's V when defined. For sparse 2×2 tables, use Fisher's exact test; flag unreliable asymptotic tests for larger sparse tables.
- With strata and explicit binary exposure/outcome labels, compare the pooled outcome-rate difference with each stratum's difference, using the same direction throughout.
- Flag a Simpson reversal only when all evaluable strata share one nonzero direction and the pooled difference has the opposite sign. Report mixed directions, missing groups, and sparse strata separately.
- Return pooled/stratum rates, denominators, differences, and a brief explanation of changing group composition. Describe the reversal as an observed pattern, not proof of confounding or causality.
- General categorical tables remain supported; automatic Simpson (paradox) detection when a dimention is binary.

### 5. `classification_metrics`

- Scope: binary classification. Input: true labels, explicit positive label, and predicted labels and/or positive-class probabilities.
- Return precision, recall, accuracy, F1, confusion matrix, support, and ROC AUC. AUC uses probabilities, never thresholded predictions; return `null` without scores or both classes.
- For probabilities, use `score >= threshold`, defaulting to 0.5. If labels and scores are both supplied, labels determine reported classification metrics; scores determine AUC.
- Suggest a threshold only from a separately supplied validation set of labels/probabilities. Maximize F1 by default, optionally Youden's J; break ties by choosing the highest threshold.
- Return the objective, selected threshold, and validation metrics; evaluate that threshold on the evaluation set separately. Never silently tune on the reported evaluation data.
- Explain undefined precision/recall/F1 with `null` and a reason. Report class imbalance and clarify that a suggested threshold depends on the chosen objective.

## Delivery checklist

- [ ] Create `pyproject.toml`, lockfile, server entrypoint, schemas, and small statistical modules using Prophet, pandas, SciPy, and scikit-learn.
- [ ] Implement the five tools with shared validation and concise outputs.
- [ ] Add focused tests: known correlations/contingencies, a Simpson reversal and non-reversal, classification edge cases and threshold leakage prevention, and a synthetic Prophet fit/next forecast.
- [ ] Add `Dockerfile` and `.dockerignore`; verify non-root startup and fitting without runtime network access.
- [ ] Smoke-test MCP initialize, tool discovery, successful/error calls, and clean shutdown through both HTTP and stdio using the pinned SDK client and an independent MCP client.
- [ ] Write a short README with Docker build/run commands, HTTP/stdio client configuration, one example per tool, limits, and statistical assumptions.

## Reference docs

- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Prophet coefficient interpretation](https://facebook.github.io/prophet/api/prophet/utilities.html)
- [SciPy statistical functions](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [scikit-learn classification metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html)
- [Threshold tuning and data separation](https://scikit-learn.org/stable/modules/classification_threshold.html)
