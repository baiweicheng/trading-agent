# Requirements Document

## Introduction

This specification defines the first usable Phase 1 vertical slice of the Quantitative Research Platform. The slice provides a complete local research loop for a single developer: ingest daily US-equities data from a free source, validate and preserve reproducible data snapshots, execute one interpretable long-only momentum baseline with Zipline Reloaded, compare results with SPY, track immutable experiments and artifacts locally, and operate the workflow through a visual Web UI.

The slice is intentionally narrow. The initial strategy universe is an explicit configurable ticker list rather than an index-membership dataset. The initial source is yfinance behind a narrow provider interface. Free Yahoo Finance data does not guarantee point-in-time universe membership, institutional data quality, complete records, or elimination of survivorship bias. Every relevant dataset and result must disclose those limitations together with the configured data-quality, cost, and execution assumptions.

Later Phase 1 expansion and roadmap Phases 2–4 are Future_Scope. This specification excludes alpha-factor libraries beyond the baseline, machine learning, portfolio optimization, LLM functionality, distributed jobs, network APIs such as FastAPI, broker integration, paper trading, and live execution. Alphalens Reloaded and Pyfolio Reloaded are not required unless a later specification adds capabilities that need those libraries. No implementation or scaffolding is part of this specification phase.

## Glossary

- **Phase_1_Platform**: The complete local software system defined by this requirements document.
- **Application_Services**: Framework-independent Python operations that coordinate ingestion, snapshots, backtests, evaluation, and experiment tracking.
- **Web_UI**: The Streamlit visual interface that invokes Application_Services.
- **Python_3_11**: The supported Python runtime line, `>=3.11,<3.12`.
- **Project_Package**: The installable Python project whose build metadata and dependencies are declared in `pyproject.toml`.
- **Project_Root**: The directory containing the Project_Package `pyproject.toml`.
- **Future_Scope**: Capabilities intentionally deferred beyond this first Phase 1 vertical slice.
- **Configuration**: Settings available from documented defaults, one YAML_Document, and explicitly mapped Environment_Variables.
- **Resolved_Configuration**: The fully merged and validated Configuration used by an operation.
- **Non_Secret_Configuration**: The projection of Resolved_Configuration that preserves non-secret values and Secret presence markers but excludes Secret values.
- **Configuration_Manager**: The component that parses, merges, validates, resolves paths, and redacts Configuration.
- **Configuration_Serializer**: The component that writes a deterministic, canonical, human-readable YAML representation of Non_Secret_Configuration.
- **Configuration_Schema**: The Pydantic model that defines allowed settings, types, required values, defaults, and bounds.
- **YAML_Document**: A configuration file conforming to the YAML syntax accepted by the selected safe YAML parser.
- **Environment_Variable**: A process-level key-value override for an explicitly mapped Configuration field.
- **Secret**: A credential or sensitive value supplied outside tracked source and excluded from durable or displayed output.
- **Redaction_Marker**: The literal value `[REDACTED]`, which represents a Secret without exposing any Secret characters.
- **Requested_Date_Range**: An inclusive pair of ISO 8601 calendar dates whose start date is not later than the end date.
- **Default_Universe**: The initial explicit strategy ticker list `AAPL`, `JPM`, `MSFT`, `PG`, and `XOM`.
- **Configured_Universe**: A resolved, ordered, duplicate-free list of 1–25 normalized US-equity ticker symbols; SPY is tracked separately as Benchmark_Symbol unless SPY is explicitly included in Configured_Universe.
- **Benchmark_Symbol**: The ticker `SPY`, used for baseline comparison and not automatically treated as a strategy candidate.
- **Market_Data_Provider**: The narrow application-facing contract for requesting daily market records by symbols and Requested_Date_Range.
- **YFinance_Adapter**: The initial Market_Data_Provider implementation that obtains free Yahoo Finance data through yfinance.
- **Data_Ingestion_Service**: The Application_Service that obtains, stages, normalizes, validates, and publishes market data.
- **Retry_Policy**: Configuration containing a bounded attempt count, initial delay, maximum delay, and backoff multiplier.
- **Retryable_Failure**: A transient provider failure category for which another request attempt is permitted by Retry_Policy.
- **Terminal_Failure**: A provider failure category for which another request attempt with unchanged input is not permitted.
- **Rate_Conscious_Batch**: A configured group of 1–10 distinct symbols requested together.
- **Write_Chunk_Size**: The configured maximum number of normalized rows persisted in one sequential write chunk.
- **Provider_Record**: One unmodified logical record returned by YFinance_Adapter, including available provider fields and provenance.
- **Raw_Dataset**: The append-only representation of Provider_Records before normalization.
- **Daily_Bar**: One normalized daily observation for one symbol and one Exchange_Session.
- **Exchange_Session**: A trading session identified by the pinned official NYSE exchange calendar rather than by a calendar-day sequence.
- **Session_Key**: The pair of normalized symbol and Exchange_Session that uniquely identifies a Daily_Bar.
- **UTC_Timestamp**: A timezone-aware timestamp represented in Coordinated Universal Time.
- **Normalizer**: The component that deterministically converts Provider_Records into Normalized_Dataset candidates.
- **Normalized_Dataset**: Accepted Daily_Bars with canonical symbols, Exchange_Sessions, UTC_Timestamps, adjusted OHLCV fields, source provenance, and validation status.
- **Adjusted_OHLCV**: Open, high, low, close, and volume values transformed under the declared Corporate_Action_Policy.
- **Adjusted_Open_Price**: The open value from Adjusted_OHLCV used as the base price for Next_Session_Execution.
- **Corporate_Action**: A provider-reported split or dividend event associated with a symbol and Exchange_Session.
- **Corporate_Action_Policy**: The versioned algorithm and provider fields used to derive Adjusted_OHLCV and represent Corporate_Actions.
- **Validation_Service**: The component that applies deterministic row, key, session, and dataset integrity checks.
- **Quarantine**: Append-only storage for rejected records together with reason codes, source identity, offending values, and detection time.
- **Data_Gap**: An expected Exchange_Session within a requested range for which no accepted Daily_Bar exists for a requested symbol.
- **Staleness_Threshold**: A configured non-negative count of Exchange_Sessions allowed between the latest expected session and latest accepted Daily_Bar.
- **Partial_Success**: A completed ingestion in which usable data is published while at least one requested symbol has a failure, Quarantine record, Data_Gap, or stale status.
- **Parquet_Store**: Partitioned Parquet files used for Raw_Dataset, Normalized_Dataset, Quarantine, and tabular Artifacts.
- **Metadata_Store**: A local DuckDB database containing dataset, snapshot, Run, validation, job, and Artifact indexes.
- **Data_Snapshot**: An immutable, versioned view of accepted market data and associated validation records used by a Run.
- **Snapshot_Manager**: The component that atomically creates, verifies, lists, and resolves Data_Snapshots.
- **Snapshot_ID**: A content-derived identifier computed from canonical Content_Identity fields and referenced Checksums.
- **Manifest**: A canonical immutable record describing a Data_Snapshot or Run, referenced content, Checksums, and separate operational metadata.
- **Content_Identity**: The deterministic scientific-content projection of a Manifest; Content_Identity excludes volatile timestamps, runtime-generated identifiers, and local storage locations.
- **Checksum**: A SHA-256 digest used to verify content identity and integrity.
- **Incremental_Update**: Ingestion that starts from a prior Data_Snapshot, requests later sessions plus configured Revision_Overlap, and resolves to an existing or new Data_Snapshot.
- **Revision_Overlap**: A configured count of previously stored Exchange_Sessions requested again to detect provider revisions.
- **Backtest_Service**: The Application_Service that executes Baseline_Strategy with Zipline Reloaded.
- **Baseline_Strategy**: The monthly rebalanced, cross-sectional, long-only momentum strategy defined by this document.
- **Signal_Session**: The last Exchange_Session of a calendar month, after whose close Baseline_Strategy computes scores.
- **Momentum_Score**: The adjusted-close return from 252 Exchange_Sessions before a Signal_Session to 21 Exchange_Sessions before the same Signal_Session.
- **Warm_Up_Period**: The 253 Exchange_Sessions preceding the first eligible Signal_Session during which no Orders are created.
- **Eligible_Symbol**: A Configured_Universe symbol having accepted adjusted-close observations at both Momentum_Score endpoints and a tradable asset record on Signal_Session.
- **Position_Count**: A configured positive maximum number of Eligible_Symbols selected at each rebalance.
- **Strategy_Decision**: A record of one symbol's score inputs, eligibility, Momentum_Score, deterministic rank, target weight, and exclusion reason for a Signal_Session.
- **Order**: A requested whole-share change from current holdings toward a target long-only portfolio weight.
- **Next_Session_Execution**: Order simulation on the first tradable Exchange_Session after the corresponding Signal_Session, using Adjusted_Open_Price as the fill-price base.
- **Whole_Share**: An integer quantity of one equity share.
- **Commission_Model**: A configured non-negative transaction cost in basis points of traded notional, with a default of 5 basis points.
- **Slippage_Model**: A configured non-negative adverse fill adjustment in basis points of traded notional, with a default of 10 basis points.
- **Initial_Portfolio_Equity**: The fixed starting Portfolio_Equity of USD 100,000.
- **Cash_Balance**: Uninvested portfolio value, earning a 0 percent return in this slice.
- **Portfolio_Equity**: Cash_Balance plus the marked value of all open positions.
- **Leverage**: Gross position value divided by Portfolio_Equity.
- **Core_Backtest_Output**: Orders, fills, positions, Cash_Balance, daily returns, Portfolio_Equity, Strategy_Decisions, and evaluation metrics.
- **No_Look_Ahead**: The property that a decision, Order, fill, or valuation depends only on information available by the applicable decision, execution, or valuation time.
- **Stable_Rerun**: A rerun with identical Data_Snapshot, Resolved_Configuration, source revision, dependency versions, platform fingerprint, and deterministic seed that produces identical Core_Backtest_Output and scientific-content Checksums.
- **Benchmark_Series**: SPY buy-and-hold daily returns derived from the same Data_Snapshot and evaluation Exchange_Sessions as a Run.
- **Evaluation_Service**: The Application_Service that computes strategy and benchmark metrics and comparison outputs.
- **Evaluation_Metrics**: Total return, compound annual growth rate, annualized volatility, zero-risk-free-rate Sharpe ratio, maximum drawdown, turnover, total commissions, and total slippage.
- **Experiment_Tracker**: The local MLflow-backed component that records Runs, inputs, lifecycle states, metrics, and Artifacts.
- **Run**: One recorded execution of Baseline_Strategy against one pinned Data_Snapshot and one Resolved_Configuration.
- **Run_ID**: The unique operational identifier assigned to a Run before backtest execution begins.
- **Terminal_Run**: A Run in `succeeded` or `failed` state.
- **Environment_Fingerprint**: Python version, operating-system and architecture values, installed dependency versions, source revision, source dirty-state indicator, and deterministic seed recorded for a Run.
- **Artifact**: A checksummed local file produced or referenced by ingestion, backtesting, or evaluation.
- **Artifact_Store**: Append-only local storage for Data_Snapshot and Run Artifacts.
- **Comparison_Set**: An ordered selection of 2–10 successful Runs for one multi-Run comparison.
- **Job_Manager**: The component that executes one local operation synchronously and records progress and terminal state.
- **Job_State**: One of `not_started`, `running`, `succeeded`, `partially_succeeded`, or `failed`.
- **Actionable_Error**: A sanitized error containing the failed operation, cause category, affected field, input, or symbol, and a corrective action.
- **Ordinary_Table_View**: A paginated Web_UI table display, excluding explicit full-Artifact downloads.
- **Limitation_Disclosure**: A visible statement covering free-source status, point-in-time-membership and survivorship-bias limitations, provider quality and completeness limitations, explicit-universe scope, recorded data failures, and configured cost and execution assumptions.
- **Test_Suite**: Pytest tests, including Hypothesis properties for variable-input application logic and representative integration tests for external boundaries.

## Requirements

### Requirement 1: Local Phase 1 Product Boundary

**User Story:** As a solo quantitative researcher, I want a focused local research platform, so that I can complete a useful research loop without operating unnecessary infrastructure.

#### Acceptance Criteria

1. THE Phase_1_Platform SHALL run locally under Python_3_11.
2. THE Project_Package SHALL declare the supported Python range as `>=3.11,<3.12` in `pyproject.toml`.
3. THE Project_Package SHALL declare build metadata in `pyproject.toml`.
4. THE Project_Package SHALL declare runtime dependencies in `pyproject.toml`.
5. THE Project_Package SHALL declare development dependencies in `pyproject.toml`.
6. THE Phase_1_Platform SHALL use free data sources.
7. THE Phase_1_Platform SHALL use free open-source runtime dependencies.
8. THE Application_Services SHALL provide ingestion, validation, snapshot management, backtesting, experiment tracking, evaluation, comparison, and Artifact-inspection operations.
9. THE Web_UI SHALL invoke research behavior through Application_Services.
10. THE Phase_1_Platform SHALL limit executable capabilities to the first Phase 1 research loop defined in this document.
11. THE Phase_1_Platform SHALL identify later Phase 1 expansion and roadmap Phases 2–4 as Future_Scope.

### Requirement 2: Validated and Reproducible Configuration

**User Story:** As a researcher, I want validated configuration with deterministic precedence, so that each workflow is configurable and reproducible without source changes.

#### Acceptance Criteria

1. THE Configuration_Schema SHALL use Pydantic to define allowed fields, field types, required values, defaults, and bounds.
2. THE Configuration_Schema SHALL define Configured_Universe, Requested_Date_Range, local data paths, Retry_Policy, Rate_Conscious_Batch size, Staleness_Threshold, Revision_Overlap, Position_Count, Commission_Model, Slippage_Model, Write_Chunk_Size, Web_UI page size, deterministic seed, and Secret fields.
3. WHEN a YAML_Document is supplied, THE Configuration_Manager SHALL parse the YAML_Document with a safe YAML parser.
4. IF a supplied YAML_Document is syntactically invalid, THEN THE Configuration_Manager SHALL return an Actionable_Error containing the parser location and cause category.
5. IF a supplied YAML_Document has a root value other than a mapping, THEN THE Configuration_Manager SHALL return an Actionable_Error containing the root value type and the required mapping type.
6. IF a YAML_Document contains a duplicate key at any nesting level, THEN THE Configuration_Manager SHALL return an Actionable_Error containing the duplicated key path.
7. IF a YAML_Document contains an unknown key at any nesting level, THEN THE Configuration_Manager SHALL return an Actionable_Error containing the key path and allowed sibling keys.
8. THE Configuration_Manager SHALL read Environment_Variables only through an explicit mapping from Environment_Variable names to Configuration_Schema field paths.
9. IF an Environment_Variable is presented as a Configuration override without an explicit field mapping, THEN THE Configuration_Manager SHALL return an Actionable_Error identifying the Environment_Variable name.
10. THE Configuration_Manager SHALL resolve each Configuration_Schema field in ascending precedence order of documented default, YAML_Document value, and explicitly mapped Environment_Variable value.
11. WHEN more than one Configuration source supplies the same field path, THE Configuration_Manager SHALL retain the value from the highest-precedence source.
12. WHEN a Configuration source supplies a nested mapping, THE Configuration_Manager SHALL apply precedence independently to each Configuration_Schema leaf field.
13. WHEN Configuration resolution begins, THE Configuration_Manager SHALL validate every resolved field and cross-field constraint with Configuration_Schema before returning Resolved_Configuration.
14. IF any Configuration validation fails, THEN THE Configuration_Manager SHALL return an Actionable_Error before an Application_Services operation begins.
15. IF multiple Configuration validations fail, THEN THE Configuration_Manager SHALL return one Actionable_Error per failing field in Configuration_Schema field order.
16. IF a required Configuration field remains absent after precedence resolution, THEN THE Configuration_Manager SHALL return an Actionable_Error containing the field path and required status.
17. IF a Configuration field has an invalid type, THEN THE Configuration_Manager SHALL return an Actionable_Error containing the field path and accepted type.
18. IF a Configuration field falls outside a declared bound, THEN THE Configuration_Manager SHALL return an Actionable_Error containing the field path and accepted bound.
19. THE Configuration_Manager SHALL remove surrounding whitespace from each Configured_Universe symbol.
20. THE Configuration_Manager SHALL convert letters in each Configured_Universe symbol to uppercase.
21. THE Configuration_Manager SHALL preserve the supplied order of distinct normalized Configured_Universe symbols.
22. IF Configured_Universe contains an empty normalized symbol, THEN THE Configuration_Manager SHALL return an Actionable_Error identifying the list position.
23. IF Configured_Universe contains a duplicate normalized symbol, THEN THE Configuration_Manager SHALL return an Actionable_Error identifying the duplicate symbol.
24. THE Configuration_Schema SHALL bound Configured_Universe length from 1 through 25 symbols.
25. WHEN no Configured_Universe value is supplied by YAML_Document or mapped Environment_Variable, THE Configuration_Manager SHALL apply Default_Universe.
26. THE Configuration_Schema SHALL require each Requested_Date_Range value to be an ISO 8601 calendar date.
27. THE Configuration_Schema SHALL require the Requested_Date_Range start date to be no later than the end date.
28. THE Configuration_Schema SHALL bound Retry_Policy total attempt count from 1 through 5 attempts.
29. THE Configuration_Schema SHALL bound Retry_Policy initial delay from 0 through 60 seconds.
30. THE Configuration_Schema SHALL bound Retry_Policy maximum delay from the configured initial delay through 60 seconds.
31. THE Configuration_Schema SHALL bound Retry_Policy backoff multiplier from 1.0 through 4.0.
32. WHEN no Retry_Policy attempt-count override is supplied, THE Configuration_Manager SHALL apply 3 total attempts.
33. WHEN no Retry_Policy initial-delay override is supplied, THE Configuration_Manager SHALL apply 1 second.
34. WHEN no Retry_Policy maximum-delay override is supplied, THE Configuration_Manager SHALL apply 8 seconds.
35. WHEN no Retry_Policy backoff-multiplier override is supplied, THE Configuration_Manager SHALL apply 2.0.
36. THE Configuration_Schema SHALL bound Rate_Conscious_Batch size from 1 through 10 symbols.
37. WHEN no Rate_Conscious_Batch size override is supplied, THE Configuration_Manager SHALL apply 5 symbols.
38. THE Configuration_Schema SHALL bound Staleness_Threshold from 0 through 252 Exchange_Sessions.
39. WHEN no Staleness_Threshold override is supplied, THE Configuration_Manager SHALL apply 1 Exchange_Session.
40. THE Configuration_Schema SHALL bound Revision_Overlap from 0 through 252 Exchange_Sessions.
41. WHEN no Revision_Overlap override is supplied, THE Configuration_Manager SHALL apply 5 Exchange_Sessions.
42. THE Configuration_Schema SHALL bound Position_Count from 1 through the Configured_Universe symbol count.
43. WHEN no Position_Count override is supplied, THE Configuration_Manager SHALL apply the smaller of 5 and the Configured_Universe symbol count.
44. THE Configuration_Schema SHALL require Commission_Model to contain a finite non-negative basis-point value.
45. WHEN no Commission_Model override is supplied, THE Configuration_Manager SHALL apply 5 basis points.
46. THE Configuration_Schema SHALL require Slippage_Model to contain a finite non-negative basis-point value.
47. WHEN no Slippage_Model override is supplied, THE Configuration_Manager SHALL apply 10 basis points.
48. THE Configuration_Schema SHALL bound the deterministic seed from 0 through 4,294,967,295.
49. WHEN no deterministic-seed override is supplied, THE Configuration_Manager SHALL apply 0.
50. THE Configuration_Schema SHALL bound Write_Chunk_Size from 1 through 100,000 rows.
51. WHEN no Write_Chunk_Size override is supplied, THE Configuration_Manager SHALL apply 50,000 rows.
52. THE Configuration_Schema SHALL bound Web_UI page size from 1 through 100 rows.
53. WHEN no Web_UI page-size override is supplied, THE Configuration_Manager SHALL apply 100 rows.
54. THE Configuration_Manager SHALL use the Project_Root containing the Project_Package `pyproject.toml` as the sole base directory for relative configured local paths.
55. IF Project_Root cannot be resolved to exactly one directory containing the Project_Package `pyproject.toml`, THEN THE Configuration_Manager SHALL return an Actionable_Error identifying the `pyproject.toml` boundary.
56. WHEN a configured local path is relative, THE Configuration_Manager SHALL normalize the path against Project_Root.
57. IF a normalized relative configured local path resolves outside Project_Root, THEN THE Configuration_Manager SHALL return an Actionable_Error containing the configured field path and Project_Root boundary.
58. WHEN a configured local path is absolute, THE Configuration_Manager SHALL preserve the normalized absolute path.
59. WHEN Configuration_Serializer receives Resolved_Configuration, THE Configuration_Serializer SHALL emit every field of Non_Secret_Configuration.
60. THE Configuration_Serializer SHALL order emitted fields by Configuration_Schema field order.
61. THE Configuration_Serializer SHALL emit canonical scalar representations.
62. THE Configuration_Serializer SHALL encode canonical output as UTF-8.
63. THE Configuration_Serializer SHALL use LF (`\n`) line endings.
64. THE Configuration_Serializer SHALL terminate canonical output with one LF (`\n`) character.
65. WHEN Configuration_Serializer serializes equivalent Non_Secret_Configuration values, THE Configuration_Serializer SHALL produce byte-equivalent YAML_Documents.
66. WHEN canonical Configuration_Serializer output is parsed as the YAML_Document without mapped Environment_Variable overrides under the same Project_Root, THE Configuration_Manager SHALL reproduce an equivalent Non_Secret_Configuration.
67. WHEN Configuration_Serializer encounters a resolved Secret field, THE Configuration_Serializer SHALL emit Redaction_Marker as the complete field value.
68. WHEN Configuration_Manager parses Redaction_Marker for a Secret field in canonical output, THE Configuration_Manager SHALL preserve the corresponding Secret field as unresolved.
69. WHEN an Actionable_Error refers to a Secret field value, THE Configuration_Manager SHALL substitute Redaction_Marker for the Secret field value.
70. WHEN any Phase_1_Platform component records Resolved_Configuration in a durable Artifact, THE Phase_1_Platform SHALL record Non_Secret_Configuration.
71. WHEN Resolved_Configuration is recorded in Metadata_Store, THE Configuration_Manager SHALL provide Non_Secret_Configuration.
72. WHEN Resolved_Configuration is displayed, THE Configuration_Manager SHALL provide Non_Secret_Configuration.

### Requirement 3: Explicit Universe and Free-Source Acquisition

**User Story:** As a researcher, I want daily data for a small explicit US-equity universe from one free source, so that I can begin research with transparent data limitations.

#### Acceptance Criteria

1. THE Data_Ingestion_Service SHALL request each distinct symbol in Configured_Universe and Benchmark_Symbol independently of current or historical index membership.
2. THE YFinance_Adapter SHALL implement Market_Data_Provider for daily US-equity data.
3. THE Data_Ingestion_Service SHALL access yfinance only through Market_Data_Provider.
4. WHEN ingestion begins, THE Data_Ingestion_Service SHALL divide requested symbols into Rate_Conscious_Batches containing 1–10 symbols.
5. WHEN a provider response fails, THE YFinance_Adapter SHALL classify the failure as Retryable_Failure or Terminal_Failure.
6. IF a Retryable_Failure occurs before Retry_Policy attempt exhaustion, THEN THE YFinance_Adapter SHALL schedule another attempt using the configured capped backoff.
7. IF a Retryable_Failure reaches Retry_Policy attempt exhaustion, THEN THE YFinance_Adapter SHALL return one Actionable_Error for every affected symbol.
8. IF a Terminal_Failure occurs, THEN THE YFinance_Adapter SHALL stop attempts for the unchanged request and return one Actionable_Error for every affected symbol.
9. THE YFinance_Adapter SHALL perform no more provider attempts than Retry_Policy attempt count for one symbol batch.
10. WHEN a provider response identifies per-symbol outcomes, THE Data_Ingestion_Service SHALL preserve each symbol outcome independently of other symbols in the same Rate_Conscious_Batch.
11. WHEN some requested symbols succeed and other requested symbols fail, THE Data_Ingestion_Service SHALL classify ingestion as Partial_Success.
12. WHEN Provider_Records are received, THE Data_Ingestion_Service SHALL preserve available unmodified provider fields in Raw_Dataset.
13. THE Data_Ingestion_Service SHALL record provider name for each request.
14. THE Data_Ingestion_Service SHALL record retrieval UTC_Timestamp for each request.
15. THE Data_Ingestion_Service SHALL record requested symbols for each request.
16. THE Data_Ingestion_Service SHALL record Requested_Date_Range for each request.
17. THE Data_Ingestion_Service SHALL record provider response status for each request.
18. IF the provider returns no Provider_Records for a requested symbol and Requested_Date_Range, THEN THE Data_Ingestion_Service SHALL return an Actionable_Error identifying the symbol and Requested_Date_Range.
19. THE Data_Ingestion_Service SHALL attach Limitation_Disclosure to every Data_Snapshot.

### Requirement 4: Session-Aware Normalization and Corporate Actions

**User Story:** As a researcher, I want canonical daily bars with declared adjustment rules, so that strategy inputs have consistent meaning across runs.

#### Acceptance Criteria

1. WHEN a Provider_Record maps to an Exchange_Session, THE Normalizer SHALL produce a Daily_Bar candidate keyed by normalized symbol and Exchange_Session.
2. IF a provider date does not map to an Exchange_Session, THEN THE Validation_Service SHALL place the Provider_Record in Quarantine with a non-session reason code.
3. THE Normalizer SHALL represent stored event times as UTC_Timestamps.
4. THE Normalizer SHALL preserve Exchange_Session date separately from stored event times.
5. THE Normalizer SHALL preserve each available raw price field from Provider_Record.
6. THE Normalizer SHALL preserve each available raw volume field from Provider_Record.
7. THE Normalizer SHALL preserve each available split field from Provider_Record.
8. THE Normalizer SHALL preserve each available dividend field from Provider_Record.
9. THE Normalizer SHALL preserve available provider metadata from Provider_Record.
10. THE Normalizer SHALL produce Adjusted_OHLCV under one versioned Corporate_Action_Policy.
11. THE Corporate_Action_Policy SHALL declare source fields and adjustment equations.
12. THE Corporate_Action_Policy SHALL declare split, dividend, volume, and rounding treatment.
13. WHEN a Corporate_Action is present, THE Normalizer SHALL preserve the provider-reported Corporate_Action alongside adjusted and raw fields.
14. IF a market observation is absent, THEN THE Normalizer SHALL produce zero Daily_Bar candidates for the absent observation.
15. IF a market observation is absent, THEN THE Validation_Service SHALL represent the absent observation as Data_Gap.
16. WHEN identical Raw_Dataset content and Resolved_Configuration are normalized more than once, THE Normalizer SHALL produce byte-equivalent canonical Daily_Bar candidates.
17. WHEN provider-record order changes without changing Provider_Record content, THE Normalizer SHALL produce the same sorted Daily_Bar candidates.
18. WHEN Rate_Conscious_Batch order changes without changing Provider_Record content, THE Normalizer SHALL produce the same sorted Daily_Bar candidates.

### Requirement 5: Validation, Quarantine, Gaps, and Staleness

**User Story:** As a researcher, I want invalid and incomplete data made explicit, so that research results do not silently depend on fabricated or corrupt observations.

#### Acceptance Criteria

1. WHEN a Daily_Bar candidate is validated, THE Validation_Service SHALL require a non-empty normalized symbol.
2. WHEN a Daily_Bar candidate is validated, THE Validation_Service SHALL require a valid Exchange_Session.
3. WHEN a Daily_Bar candidate is validated, THE Validation_Service SHALL require finite positive open, high, low, and close values.
4. WHEN a Daily_Bar candidate is validated, THE Validation_Service SHALL require a finite non-negative volume value.
5. WHEN a Daily_Bar candidate is validated, THE Validation_Service SHALL require high to be at least open, low, and close.
6. WHEN a Daily_Bar candidate is validated, THE Validation_Service SHALL require low to be at most open, high, and close.
7. IF a Daily_Bar candidate violates a row-validation rule, THEN THE Validation_Service SHALL place the Daily_Bar candidate in Quarantine with the rule identifier and offending values.
8. WHEN duplicate Session_Keys have byte-equivalent canonical values, THE Validation_Service SHALL retain one deterministic Daily_Bar.
9. WHEN duplicate Session_Keys have byte-equivalent canonical values, THE Validation_Service SHALL record the collapsed duplicate count.
10. IF duplicate Session_Keys contain conflicting canonical values, THEN THE Validation_Service SHALL place every Daily_Bar candidate for the conflicting Session_Key in Quarantine.
11. IF duplicate Session_Keys contain conflicting canonical values, THEN THE Validation_Service SHALL produce zero accepted Daily_Bars for the conflicting Session_Key.
12. THE Validation_Service SHALL ensure that Normalized_Dataset contains exactly one accepted Daily_Bar per Session_Key.
13. THE Validation_Service SHALL ensure that every accepted Daily_Bar traces to a Provider_Record preserved in Raw_Dataset.
14. WHEN an expected Exchange_Session lacks an accepted Daily_Bar, THE Validation_Service SHALL record a Data_Gap containing symbol and Exchange_Session.
15. WHEN an expected Exchange_Session lacks an accepted Daily_Bar, THE Validation_Service SHALL produce zero accepted Daily_Bars for the missing Session_Key.
16. WHEN the latest accepted Daily_Bar trails the latest expected Exchange_Session by more than Staleness_Threshold, THE Validation_Service SHALL mark the symbol stale.
17. WHEN a symbol is stale, THE Validation_Service SHALL record the lag in Exchange_Sessions.
18. THE Validation_Service SHALL produce a validation report containing accepted counts, quarantined counts by reason, duplicate counts, Data_Gaps, stale symbols, failed symbols, and covered session ranges.
19. WHEN any requested symbol has a failure, Quarantine record, Data_Gap, or stale status, THE Data_Ingestion_Service SHALL surface Partial_Success while preserving successfully published data.
20. IF Benchmark_Symbol has a Data_Gap in a requested evaluation range, THEN THE Validation_Service SHALL mark the Data_Snapshot as not ready for benchmark comparison over that range.
21. WHEN validation is repeated over identical canonical input, THE Validation_Service SHALL produce the same accepted Daily_Bars.
22. WHEN validation is repeated over identical canonical input, THE Validation_Service SHALL produce the same Quarantine decisions.
23. WHEN validation is repeated over identical canonical input, THE Validation_Service SHALL produce the same validation-report Content_Identity.

### Requirement 6: Versioned Data Storage and Immutable Snapshots

**User Story:** As a researcher, I want checksummed data snapshots, so that every experiment can identify and verify its exact market data.

#### Acceptance Criteria

1. THE Parquet_Store SHALL store Raw_Dataset and Normalized_Dataset content in separate partitioned Parquet collections.
2. THE Parquet_Store SHALL partition Normalized_Dataset content by symbol and Exchange_Session year.
3. THE Metadata_Store SHALL index provider requests, partitions, validation results, Data_Snapshots, Runs, jobs, and Artifacts in DuckDB.
4. WHEN a Data_Snapshot is created, THE Snapshot_Manager SHALL create a Manifest containing Snapshot_ID, parent Snapshot_ID when applicable, provider provenance, requested range, covered range, Configured_Universe, Benchmark_Symbol, schema version, Corporate_Action_Policy version, NYSE calendar version, Configuration Checksum, partition Checksums, row counts, validation summary, and creation UTC_Timestamp.
5. THE Snapshot_Manager SHALL calculate a Checksum for every Parquet partition referenced by a Data_Snapshot Manifest.
6. THE Snapshot_Manager SHALL calculate a Checksum for every validation Artifact referenced by a Data_Snapshot Manifest.
7. THE Snapshot_Manager SHALL derive Snapshot_ID from canonical Content_Identity and referenced Checksums.
8. THE Snapshot_Manager SHALL exclude creation, retrieval, detection, and job-progress timestamps from Snapshot_ID calculation.
9. THE Snapshot_Manager SHALL retain excluded timestamps as operational metadata outside Content_Identity.
10. WHEN a Data_Snapshot is published, THE Snapshot_Manager SHALL make the Data_Snapshot Manifest immutable through Phase_1_Platform operations.
11. WHEN a Data_Snapshot is published, THE Snapshot_Manager SHALL make referenced Data_Snapshot content immutable through Phase_1_Platform operations.
12. WHEN a Data_Snapshot is opened, THE Snapshot_Manager SHALL verify every referenced Checksum before use.
13. IF referenced Data_Snapshot content fails Checksum verification, THEN THE Snapshot_Manager SHALL reject the Data_Snapshot with an Actionable_Error.
14. WHEN a Run starts, THE Backtest_Service SHALL pin exactly one Snapshot_ID before reading market data.
15. WHILE a Run is active, THE Backtest_Service SHALL read market data only from the pinned Data_Snapshot.
16. IF a later ingestion or verification operation fails, THEN THE Snapshot_Manager SHALL preserve every previously published valid Data_Snapshot.
17. WHEN a Manifest and referenced content are copied to another valid local storage location, THE Snapshot_Manager SHALL resolve the copied content to the same Snapshot_ID after verification.
18. IF an operation attempts to mutate a published Data_Snapshot, THEN THE Snapshot_Manager SHALL reject the operation with an Actionable_Error and require publication of a new Data_Snapshot.

### Requirement 7: Atomic, Incremental, and Idempotent Ingestion

**User Story:** As a researcher, I want safe repeatable data updates, so that retries and provider revisions cannot corrupt prior research inputs.

#### Acceptance Criteria

1. WHEN ingestion begins, THE Data_Ingestion_Service SHALL write new Raw_Dataset, Normalized_Dataset, Quarantine, and Manifest candidates to a staging location.
2. WHEN staging validation and Checksum generation succeed, THE Snapshot_Manager SHALL atomically publish the complete Data_Snapshot.
3. IF ingestion, normalization, validation, or Checksum generation fails before publication, THEN THE Snapshot_Manager SHALL leave the staging result unpublished.
4. IF ingestion, normalization, validation, or Checksum generation fails before publication, THEN THE Snapshot_Manager SHALL preserve all prior valid Data_Snapshots.
5. WHEN Incremental_Update starts from a prior Data_Snapshot, THE Data_Ingestion_Service SHALL request the latest Revision_Overlap stored Exchange_Sessions and all requested later Exchange_Sessions.
6. WHEN validated re-requested Provider_Records differ from prior accepted content, THE Snapshot_Manager SHALL publish revised accepted content only in a new Data_Snapshot.
7. WHEN re-requested Provider_Records match prior content and Requested_Date_Range is unchanged, THE Snapshot_Manager SHALL resolve Incremental_Update to the existing Snapshot_ID.
8. WHEN Incremental_Update resolves to an existing Snapshot_ID, THE Snapshot_Manager SHALL avoid duplicate accepted partitions.
9. WHEN an interrupted ingestion is retried with identical inputs, THE Data_Ingestion_Service SHALL produce the same published Content_Identity as uninterrupted ingestion.
10. WHEN symbol batches are processed in different orders with identical Provider_Record content, THE Snapshot_Manager SHALL produce the same Snapshot_ID.
11. IF a candidate partition path would replace content referenced by a published Data_Snapshot, THEN THE Snapshot_Manager SHALL write the candidate partition to a new content path.
12. WHEN Incremental_Update has a failed symbol with prior accepted content, THE Snapshot_Manager SHALL retain the prior accepted content for that symbol in the candidate Data_Snapshot.
13. WHEN Incremental_Update has a failed symbol without prior accepted content, THE Snapshot_Manager SHALL record zero accepted content for that symbol in the candidate Data_Snapshot.
14. WHEN Incremental_Update publishes usable content with failed symbols, THE Snapshot_Manager SHALL record failed symbols and retained prior coverage in the new Manifest.
15. WHEN Incremental_Update publishes usable content with failed symbols, THE Data_Ingestion_Service SHALL classify the operation as Partial_Success.
16. IF Incremental_Update cannot publish a valid candidate Data_Snapshot, THEN THE Snapshot_Manager SHALL keep the latest prior valid Data_Snapshot resolvable.

### Requirement 8: Interpretable Monthly Momentum Baseline

**User Story:** As a researcher, I want one simple and interpretable baseline strategy, so that the platform can demonstrate a complete research workflow before adding advanced methods.

#### Acceptance Criteria

1. THE Baseline_Strategy SHALL treat the last Exchange_Session of each calendar month as Signal_Session.
2. THE Baseline_Strategy SHALL calculate Momentum_Score from adjusted-close values 252 and 21 Exchange_Sessions before each Signal_Session.
3. WHILE Warm_Up_Period is incomplete, THE Baseline_Strategy SHALL maintain Initial_Portfolio_Equity in Cash_Balance without creating Orders.
4. WHEN either Momentum_Score endpoint is unavailable for a symbol, THE Baseline_Strategy SHALL mark the symbol ineligible with the missing endpoint reason.
5. WHEN a symbol lacks a tradable asset record on Signal_Session, THE Baseline_Strategy SHALL mark the symbol ineligible with the asset-status reason.
6. WHEN Momentum_Scores are available, THE Baseline_Strategy SHALL rank Eligible_Symbols by descending Momentum_Score.
7. WHEN Eligible_Symbols have equal Momentum_Scores, THE Baseline_Strategy SHALL break the tie by ascending normalized ticker symbol.
8. WHEN at least one Eligible_Symbol exists, THE Baseline_Strategy SHALL select the highest-ranked Eligible_Symbols up to Position_Count.
9. WHEN symbols are selected, THE Baseline_Strategy SHALL assign equal non-negative target weights to selected symbols.
10. WHEN symbols are selected, THE Baseline_Strategy SHALL make selected-symbol target weights sum to 1.0.
11. IF no Eligible_Symbol exists, THEN THE Baseline_Strategy SHALL assign 100 percent of Portfolio_Equity to Cash_Balance.
12. THE Baseline_Strategy SHALL create one Strategy_Decision for every Configured_Universe symbol on every Signal_Session.
13. THE Strategy_Decision SHALL contain score inputs, Momentum_Score, eligibility, rank, target weight, and exclusion reason.
14. THE Baseline_Strategy SHALL record Position_Count and fixed Momentum_Score endpoints in the Run Manifest.
15. WHEN identical ordered inputs and Resolved_Configuration are evaluated, THE Baseline_Strategy SHALL produce identical Strategy_Decisions.

### Requirement 9: Bias-Controlled Zipline Backtesting and Portfolio Accounting

**User Story:** As a researcher, I want conservative session-aware backtests, so that baseline results avoid same-bar execution, look-ahead, shorting, and leverage.

#### Acceptance Criteria

1. THE Backtest_Service SHALL execute Baseline_Strategy portfolio simulation with Zipline Reloaded.
2. THE Backtest_Service SHALL initialize each Run with Initial_Portfolio_Equity of USD 100,000 in Cash_Balance.
3. WHEN a Signal_Session closes, THE Backtest_Service SHALL create resulting Orders for Next_Session_Execution.
4. THE Backtest_Service SHALL base each fill on Adjusted_Open_Price from the first tradable Exchange_Session after the corresponding Signal_Session.
5. THE Backtest_Service SHALL exclude Signal_Session prices from fill-price calculation for Orders created after Signal_Session close.
6. THE Backtest_Service SHALL express every Order quantity and fill quantity as Whole_Shares.
7. WHEN a buy quantity would violate Cash_Balance or Leverage constraints, THE Backtest_Service SHALL reduce the buy quantity to the greatest permitted non-negative Whole_Share quantity.
8. IF no valid Adjusted_Open_Price exists on the first Exchange_Session after Signal_Session for an Order, THEN THE Backtest_Service SHALL leave the Order unfilled.
9. WHEN an Order remains unfilled, THE Backtest_Service SHALL record an Actionable_Error identifying the symbol and execution Exchange_Session.
10. THE Backtest_Service SHALL apply Commission_Model to traded notional for each simulated fill.
11. THE Backtest_Service SHALL increase buy fill prices and decrease sell fill prices according to Slippage_Model.
12. THE Backtest_Service SHALL maintain non-negative position quantities for every symbol after each fill.
13. THE Backtest_Service SHALL maintain Cash_Balance at or above USD 0.00 after each fill and cost application.
14. THE Backtest_Service SHALL maintain Leverage from 0.0 through 1.0 after each fill.
15. THE Backtest_Service SHALL apply a 0 percent return to Cash_Balance.
16. THE Backtest_Service SHALL process Exchange_Sessions according to the NYSE calendar version pinned by Data_Snapshot.
17. THE Backtest_Service SHALL apply the Data_Snapshot Corporate_Action_Policy consistently to positions and valuation.
18. WHEN Portfolio_Equity is recorded, THE Backtest_Service SHALL make Portfolio_Equity equal Cash_Balance plus marked position values within USD 0.01.
19. WHEN data after a Signal_Session changes without changing data available by Signal_Session close, THE Baseline_Strategy SHALL preserve Strategy_Decisions for that Signal_Session.
20. WHEN data after a fill session changes without changing data available by that fill, THE Backtest_Service SHALL preserve Orders and fills through that fill session.
21. THE Backtest_Service SHALL satisfy No_Look_Ahead for every Strategy_Decision, Order, fill, and valuation.
22. WHEN Stable_Rerun conditions hold, THE Backtest_Service SHALL produce identical Core_Backtest_Output.
23. WHEN Stable_Rerun conditions hold, THE Backtest_Service SHALL produce identical scientific-content Checksums.
24. THE Backtest_Service SHALL record the deterministic seed before execution.

### Requirement 10: SPY Evaluation and Comparable Metrics

**User Story:** As a researcher, I want baseline results evaluated against SPY on matching sessions, so that performance claims have a transparent reference.

#### Acceptance Criteria

1. WHEN Core_Backtest_Output is complete, THE Evaluation_Service SHALL construct Benchmark_Series from Benchmark_Symbol in the pinned Data_Snapshot.
2. THE Evaluation_Service SHALL align Benchmark_Series and strategy returns to identical first and last evaluation Exchange_Sessions.
3. THE Evaluation_Service SHALL use adjusted SPY values under the Data_Snapshot Corporate_Action_Policy.
4. IF Benchmark_Series has any Data_Gap in the aligned evaluation range, THEN THE Evaluation_Service SHALL block metric comparison.
5. IF Benchmark_Series has any Data_Gap in the aligned evaluation range, THEN THE Evaluation_Service SHALL return an Actionable_Error identifying every missing Exchange_Session.
6. WHEN aligned gap-free returns are available, THE Evaluation_Service SHALL calculate Evaluation_Metrics for Baseline_Strategy.
7. WHEN aligned gap-free returns are available, THE Evaluation_Service SHALL calculate Evaluation_Metrics for Benchmark_Series.
8. THE Evaluation_Service SHALL annualize return and volatility using 252 Exchange_Sessions per year.
9. THE Evaluation_Service SHALL calculate Sharpe ratio using a 0 percent risk-free rate.
10. THE Evaluation_Service SHALL report Baseline_Strategy minus Benchmark_Series differences for total return, compound annual growth rate, annualized volatility, Sharpe ratio, and maximum drawdown.
11. THE Evaluation_Service SHALL report turnover, total commissions, total slippage, unfilled Orders, and ending Cash_Balance for Baseline_Strategy.
12. THE Evaluation_Service SHALL produce a checksummed equity-curve Artifact.
13. THE Evaluation_Service SHALL produce a checksummed drawdown Artifact.
14. THE Evaluation_Service SHALL produce a checksummed monthly-return Artifact.
15. THE Evaluation_Service SHALL produce checksummed position, transaction, and Strategy_Decision Artifacts.
16. WHEN identical Core_Backtest_Output is evaluated more than once, THE Evaluation_Service SHALL produce identical Evaluation_Metrics.
17. WHEN identical Core_Backtest_Output is evaluated more than once, THE Evaluation_Service SHALL produce identical canonical tabular Artifact Checksums.
18. THE Evaluation_Service SHALL attach Limitation_Disclosure to every comparison output.

### Requirement 11: Immutable Local Experiment Tracking

**User Story:** As a researcher, I want every run and artifact tracked immutably, so that I can reproduce, inspect, and compare prior experiments.

#### Acceptance Criteria

1. THE Experiment_Tracker SHALL use a local MLflow tracking store.
2. THE Experiment_Tracker SHALL use local Artifact_Store.
3. WHEN a Run is requested, THE Experiment_Tracker SHALL assign Run_ID before Backtest_Service execution.
4. WHEN a Run is requested, THE Experiment_Tracker SHALL record `running` state before Backtest_Service execution.
5. WHEN a Run begins, THE Experiment_Tracker SHALL record Snapshot_ID, Non_Secret_Configuration, strategy identifier, strategy parameters, evaluation range, and Environment_Fingerprint.
6. THE Environment_Fingerprint SHALL include Python version, operating-system version, machine architecture, installed dependency versions, source revision, source dirty-state indicator, and deterministic seed.
7. WHEN evaluation completes without error, THE Experiment_Tracker SHALL record Evaluation_Metrics and `succeeded` state.
8. IF a Run fails, THEN THE Experiment_Tracker SHALL record `failed` state and Actionable_Error.
9. IF a Run fails after producing diagnostic Artifacts, THEN THE Experiment_Tracker SHALL preserve completed diagnostic Artifacts.
10. WHEN a Run produces Core_Backtest_Output, THE Experiment_Tracker SHALL store or reference checksummed Orders, fills, positions, returns, Portfolio_Equity, Strategy_Decisions, logs, evaluation tables, charts, and Run Manifest Artifacts.
11. THE Experiment_Tracker SHALL include Snapshot_ID and every Artifact Checksum in the Run Manifest.
12. THE Experiment_Tracker SHALL derive Run Manifest Content_Identity from deterministic scientific inputs and referenced Checksums.
13. THE Experiment_Tracker SHALL exclude Run_ID, creation time, start time, end time, and progress timestamps from Run Manifest Content_Identity.
14. THE Experiment_Tracker SHALL retain excluded Run metadata as operational metadata outside Content_Identity.
15. WHEN a Run reaches Terminal_Run state, THE Experiment_Tracker SHALL make Run inputs, metrics, lifecycle state, Run Manifest, and Artifacts immutable through Phase_1_Platform operations.
16. IF an operation attempts to mutate a Terminal_Run, THEN THE Experiment_Tracker SHALL reject the operation with an Actionable_Error and require a new Run_ID.
17. WHEN a Run Artifact is opened, THE Experiment_Tracker SHALL verify the Artifact against the Run Manifest Checksum.
18. IF a later Run fails, THEN THE Experiment_Tracker SHALL preserve prior valid Runs.
19. THE Experiment_Tracker SHALL exclude Secret values from MLflow parameters, tags, logs, Manifests, and Artifacts.
20. THE Experiment_Tracker SHALL attach Limitation_Disclosure to every Run Manifest.

### Requirement 12: Experiment Discovery and Comparison

**User Story:** As a researcher, I want to find and compare experiments, so that research accumulates instead of disappearing after each run.

#### Acceptance Criteria

1. THE Metadata_Store SHALL make Runs discoverable by Run_ID, creation time, Snapshot_ID, strategy identifier, Configured_Universe, evaluation range, and terminal state.
2. WHEN 2–10 successful Runs are selected, THE Evaluation_Service SHALL create one Comparison_Set.
3. IF fewer than 2 successful Runs are selected, THEN THE Evaluation_Service SHALL return an Actionable_Error stating the 2-Run minimum.
4. IF more than 10 Runs are selected, THEN THE Evaluation_Service SHALL return an Actionable_Error stating the 10-Run maximum.
5. WHEN a Comparison_Set is valid, THE Evaluation_Service SHALL present Evaluation_Metrics for every selected Run in one comparison table.
6. WHEN selected Runs use different Snapshot_ID values, THE Evaluation_Service SHALL display the differing Snapshot_ID values before metric comparison.
7. WHEN selected Runs use different Resolved_Configurations, THE Evaluation_Service SHALL display differing non-secret fields.
8. WHEN selected Runs use different Environment_Fingerprints, THE Evaluation_Service SHALL display differing fingerprint fields.
9. THE Evaluation_Service SHALL align selected equity curves to common Exchange_Sessions before display.
10. THE Evaluation_Service SHALL display Baseline_Strategy and Benchmark_Series equity curves for every selected Run.
11. WHEN selected Runs have different evaluation ranges, THE Evaluation_Service SHALL display each original evaluation range and the aligned comparison range.
12. WHEN a prior Run is selected, THE Experiment_Tracker SHALL expose Run Manifest, Configuration Artifact, Environment_Fingerprint, validation report, logs, and checksummed output Artifacts.
13. IF a selected Artifact fails Checksum verification, THEN THE Experiment_Tracker SHALL mark the Artifact invalid and return an Actionable_Error.
14. THE Evaluation_Service SHALL attach Limitation_Disclosure to every multi-Run comparison.

### Requirement 13: Visual End-to-End Workflow

**User Story:** As a researcher, I want to run and inspect the complete workflow in a visual Web UI, so that routine local research does not require direct script orchestration.

#### Acceptance Criteria

1. THE Web_UI SHALL provide controls for selecting a YAML_Document, Configured_Universe, Requested_Date_Range, Retry_Policy, Rate_Conscious_Batch size, Position_Count, Commission_Model, Slippage_Model, and deterministic seed.
2. WHEN Configuration values are edited, THE Web_UI SHALL validate Resolved_Configuration before enabling ingestion or backtest execution.
3. WHEN ingestion is requested, THE Web_UI SHALL invoke Data_Ingestion_Service synchronously through Application_Services.
4. WHILE ingestion is running, THE Web_UI SHALL display current Job_State, completed symbol count, total symbol count, current stage, and accumulated sanitized warnings.
5. WHEN ingestion reaches a terminal Job_State, THE Web_UI SHALL display Snapshot_ID or Actionable_Error.
6. WHEN a Data_Snapshot is selected, THE Web_UI SHALL display Manifest provenance, covered sessions, validation summary, Quarantine reasons, Data_Gaps, staleness, failed symbols, and comparison readiness.
7. WHEN a backtest is requested, THE Web_UI SHALL require explicit selection of one Checksum-verified Snapshot_ID.
8. WHEN a backtest is requested, THE Web_UI SHALL invoke Backtest_Service synchronously through Application_Services.
9. WHILE a backtest is running, THE Web_UI SHALL display current Job_State, processed Exchange_Sessions, total Exchange_Sessions, and accumulated sanitized warnings.
10. WHEN a backtest succeeds, THE Web_UI SHALL display Run_ID, Baseline_Strategy Evaluation_Metrics, Benchmark_Series Evaluation_Metrics, comparison differences, equity curves, drawdowns, positions, transactions, costs, and ending Cash_Balance.
11. WHEN a job partially succeeds or fails, THE Web_UI SHALL retain access to prior valid Data_Snapshots and Runs.
12. THE Web_UI SHALL provide Run search, Manifest inspection, validation-report inspection, log inspection, and Artifact download controls.
13. THE Web_UI SHALL allow selection of 2–10 successful Runs for multi-Run comparison.
14. IF a Web_UI multi-Run selection falls outside 2–10 Runs, THEN THE Web_UI SHALL keep comparison execution disabled and display the accepted range.
15. WHEN Web_UI displays an Ordinary_Table_View, THE Web_UI SHALL load no more than 100 rows per page.
16. WHEN Web_UI paginates an Ordinary_Table_View, THE Web_UI SHALL keep every page at or below the configured Web_UI page size.
17. WHEN a complete tabular Artifact is requested, THE Web_UI SHALL provide explicit Artifact download without loading the complete Artifact into an Ordinary_Table_View.
18. THE Web_UI SHALL display Limitation_Disclosure on every data, Data_Snapshot, Run-result, and comparison view.
19. THE Web_UI SHALL keep Limitation_Disclosure visible without requiring Artifact download.
20. THE Web_UI SHALL exclude Secret values from Resolved_Configuration displays, progress, errors, logs, and Artifacts.

### Requirement 14: Progress, Failure Isolation, and Actionable Diagnostics

**User Story:** As a researcher, I want visible progress and isolated failures, so that local jobs remain understandable and recoverable.

#### Acceptance Criteria

1. WHEN a local operation starts, THE Job_Manager SHALL transition Job_State from `not_started` to `running`.
2. WHEN a local operation completes without recorded data failures, THE Job_Manager SHALL transition Job_State to `succeeded`.
3. WHEN ingestion publishes usable data with a symbol failure, Quarantine record, Data_Gap, or stale status, THE Job_Manager SHALL transition Job_State to `partially_succeeded`.
4. IF a local operation cannot publish or record required output, THEN THE Job_Manager SHALL transition Job_State to `failed`.
5. WHILE a local operation is running, THE Job_Manager SHALL report stage name, completed work units, total work units when known, elapsed time, and sanitized warnings.
6. IF an exception crosses an Application_Services boundary, THEN THE Application_Services SHALL convert the exception into an Actionable_Error.
7. IF an exception crosses an Application_Services boundary, THEN THE Application_Services SHALL preserve sanitized diagnostic context in a local log.
8. THE Actionable_Error SHALL identify failed operation, cause category, affected field, input, or symbol when applicable, and one corrective action.
9. WHEN a symbol fails during a Rate_Conscious_Batch, THE Data_Ingestion_Service SHALL identify the failed symbol independently from successful symbols in the same Rate_Conscious_Batch.
10. WHEN a symbol fails during an update, THE Data_Ingestion_Service SHALL preserve accepted prior content for unaffected symbols.
11. IF storage publication fails, THEN THE Snapshot_Manager SHALL keep the latest valid Data_Snapshot resolvable.
12. IF experiment recording fails after Run_ID assignment, THEN THE Experiment_Tracker SHALL preserve the Run record in `failed` state with available diagnostics.
13. THE Job_Manager SHALL record progress and terminal state in Metadata_Store for later inspection.

### Requirement 15: Memory-Conscious Local Processing

**User Story:** As a researcher using an 18 GB Apple-silicon laptop, I want bounded and out-of-core processing, so that the first research loop does not require all historical data in memory.

#### Acceptance Criteria

1. WHEN requested symbols exceed one Rate_Conscious_Batch, THE Data_Ingestion_Service SHALL process successive batches containing no more than 10 symbols.
2. WHEN normalized output exceeds Write_Chunk_Size, THE Parquet_Store SHALL persist successive chunks containing no more than Write_Chunk_Size rows.
3. WHEN validation spans multiple partitions, THE Validation_Service SHALL aggregate validation results incrementally without concurrent materialization of all partitions.
4. WHEN Backtest_Service reads a Data_Snapshot, THE Backtest_Service SHALL request only configured symbols, required fields, and Exchange_Sessions for the active simulation window.
5. WHEN Evaluation_Service calculates a metric, THE Evaluation_Service SHALL read only required columns through DuckDB or partition-filtered Parquet access.
6. WHEN Web_UI displays an Ordinary_Table_View, THE Web_UI SHALL retrieve no more than 100 rows for one page.
7. THE Data_Ingestion_Service SHALL record Rate_Conscious_Batch size and Write_Chunk_Size in the Data_Snapshot Manifest.
8. THE Backtest_Service SHALL record data-window and selected-symbol bounds in the Run Manifest.

### Requirement 16: Security and Sensitive-Value Hygiene

**User Story:** As a local researcher, I want sensitive values kept out of durable outputs, so that reproducibility records do not leak credentials.

#### Acceptance Criteria

1. THE Configuration_Manager SHALL accept Secret values only from mapped Environment_Variables or ignored local secret files outside tracked source.
2. THE Project_Package SHALL identify local secret-file patterns for source-control exclusion.
3. THE Project_Package SHALL identify generated data paths for source-control exclusion.
4. WHEN a Secret value appears in an exception, provider URL, header, Configuration, or process output, THE Application_Services SHALL replace the Secret value with Redaction_Marker before logging or display.
5. WHEN Configuration_Serializer serializes a Secret field, THE Configuration_Serializer SHALL preserve the field name and replace the field value with Redaction_Marker.
6. IF Artifact metadata contains an unredacted configured Secret value, THEN THE Artifact_Store SHALL reject the Artifact metadata with an Actionable_Error.
7. THE Experiment_Tracker SHALL record Secret names or presence indicators without recording Secret values.
8. WHEN redaction is applied repeatedly, THE Configuration_Manager SHALL produce the same Redaction_Marker.
9. WHEN redaction is applied, THE Configuration_Manager SHALL reveal zero characters from the Secret value.
10. WHEN canonical Configuration containing Redaction_Marker is loaded, THE Configuration_Manager SHALL require an external Secret source before an operation that needs the Secret field.

### Requirement 17: Correctness and Integration Test Obligations

**User Story:** As a maintainer, I want automated correctness tests around critical research behavior, so that data and backtest regressions are detected before results are trusted.

#### Acceptance Criteria

1. THE Test_Suite SHALL execute with Pytest under Python_3_11.
2. THE Test_Suite SHALL include a Hypothesis property for canonical Configuration serialization and parse round trip.
3. THE Test_Suite SHALL include representative examples for Configuration precedence, unknown keys, field bounds, path resolution, and Secret redaction.
4. THE Test_Suite SHALL include a Hypothesis property for normalization determinism.
5. THE Test_Suite SHALL include a Hypothesis property for accepted Session_Key uniqueness.
6. THE Test_Suite SHALL include a Hypothesis property for ingestion idempotence.
7. THE Test_Suite SHALL include a Hypothesis property for Rate_Conscious_Batch-order confluence.
8. THE Test_Suite SHALL include Hypothesis properties for non-negative positions, non-negative Cash_Balance, Leverage bounds, Whole_Share quantities, and Portfolio_Equity accounting.
9. THE Test_Suite SHALL include a Hypothesis property for No_Look_Ahead.
10. THE Test_Suite SHALL include a Hypothesis property for Stable_Rerun output equivalence.
11. THE Test_Suite SHALL generate invalid Daily_Bar candidates to verify deterministic Quarantine reason codes.
12. THE Test_Suite SHALL generate conflicting duplicate Session_Keys to verify Quarantine of every conflicting Daily_Bar candidate.
13. THE Test_Suite SHALL generate conflicting duplicate Session_Keys to verify zero accepted Daily_Bars for each conflicting Session_Key.
14. THE Test_Suite SHALL generate missing Exchange_Sessions to verify Data_Gap reporting.
15. THE Test_Suite SHALL generate missing Exchange_Sessions to verify zero fabricated Daily_Bars.
16. THE Test_Suite SHALL verify bounded retries for Retryable_Failure.
17. THE Test_Suite SHALL verify zero repeated attempts after Terminal_Failure classification.
18. THE Test_Suite SHALL compare interrupted-and-retried ingestion with uninterrupted ingestion for equivalent published Content_Identity.
19. THE Test_Suite SHALL modify only post-decision data to verify preservation of prior Strategy_Decisions.
20. THE Test_Suite SHALL modify only post-fill data to verify preservation of prior Orders and fills.
21. THE Test_Suite SHALL verify Next_Session_Execution from Adjusted_Open_Price with configured adverse slippage and commission.
22. THE Test_Suite SHALL verify Initial_Portfolio_Equity of USD 100,000.
23. THE Test_Suite SHALL verify that published Data_Snapshot mutation attempts are rejected.
24. THE Test_Suite SHALL verify that Terminal_Run mutation attempts are rejected.
25. THE Test_Suite SHALL verify that Benchmark_Series Data_Gaps block comparison.
26. THE Test_Suite SHALL verify Comparison_Set acceptance for 2–10 successful Runs.
27. THE Test_Suite SHALL verify Comparison_Set rejection outside 2–10 Runs.
28. THE Test_Suite SHALL verify that Ordinary_Table_View pages contain no more than 100 rows.
29. THE Test_Suite SHALL use deterministic seeds and record failing Hypothesis examples.
30. THE Test_Suite SHALL use mocks or local fixtures instead of external provider calls for property-based tests.
31. WHERE external provider integration tests are enabled, THE Test_Suite SHALL limit external requests to representative smoke examples under Retry_Policy.
32. THE Test_Suite SHALL include a local end-to-end example covering Configuration parsing, fixture ingestion, validation, snapshot publication, Baseline_Strategy execution, SPY evaluation, experiment recording, and Artifact verification.
33. WHEN the local end-to-end example is rerun under Stable_Rerun conditions, THE Test_Suite SHALL verify identical Core_Backtest_Output.
34. WHEN the local end-to-end example is rerun under Stable_Rerun conditions, THE Test_Suite SHALL verify identical scientific-content Checksums.
35. THE Test_Suite SHALL include representative Web_UI examples verifying visible Limitation_Disclosure on data, result, and comparison views.

## Requirement Traceability Summary

- Requirements 1–3 establish the local, free-only, configuration-driven product and provider boundary.
- Requirements 4–7 establish deterministic normalization, validation, Quarantine, versioned storage, and atomic reproducible ingestion.
- Requirements 8–10 establish the interpretable momentum baseline, bias controls, conservative whole-share next-session execution, and SPY evaluation.
- Requirements 11–14 establish immutable experiment tracking, bounded comparison, the visual workflow, and visible failure handling.
- Requirements 15–17 establish laptop-suitable data access, Secret hygiene, and automated correctness evidence.
