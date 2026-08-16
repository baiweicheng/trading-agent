"""Immutable run-manifest assembly and publication for the local Phase 1 graph.

The evaluator owns deterministic result bytes.  This module assembles the
scientific run document around those bytes, removes operational publication
locations from its content identity, and publishes the exact canonical JSON
through the existing filesystem CAS boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from ..config.models import ResolvedConfig
from ..config.serializer import ConfigurationSerializer, non_secret_config
from ..domain.canonical import canonical_json, sha256_bytes, sha256_canonical_json
from ..domain.errors import LimitationDisclosure
from ..domain.evaluation import (
    DependencyVersion,
    EvaluationResult,
    RunContentIdentity,
    RunManifest,
    RunOperationalMetadata,
    ScientificArtifactReference,
)
from ..domain.evaluation import EnvironmentFingerprint as DomainEnvironmentFingerprint
from ..domain.execution import RunState
from ..domain.manifests import ArtifactReference, ContentAddressedObjectRef, ObjectKind
from ..domain.strategy import MomentumStrategyParameters
from .fingerprint import EnvironmentFingerprint as InfrastructureEnvironmentFingerprint
from .fingerprint import fingerprint_environment

_RUN_MANIFEST_SCHEMA_VERSION = "run_manifest_v1"
_RUN_MANIFEST_MEDIA_TYPE = "application/json"


@dataclass(frozen=True, slots=True)
class RunArtifactReference:
    """A top-level, location-aware artifact link sent to the tracker."""

    checksum: str
    role: str
    relative_uri: str
    byte_size: int
    media_type: str
    schema_version: str
    row_count: int | None
    scientific: bool = True

    @property
    def uri(self) -> str:
        """Compatibility alias used by tracker and inspection adapters."""

        return self.relative_uri


@dataclass(frozen=True, slots=True)
class RunManifestPublication:
    """Published run manifest plus the artifact links needed for finalization."""

    manifest: Mapping[str, object]
    payload: bytes
    manifest_checksum: str
    manifest_uri: str
    artifacts: tuple[RunArtifactReference, ...]
    environment_fingerprint: DomainEnvironmentFingerprint

    @property
    def checksum(self) -> str:
        """Compatibility alias for structural publication ports."""

        return self.manifest_checksum

    @property
    def uri(self) -> str:
        """Compatibility alias for structural publication ports."""

        return self.manifest_uri


class RunManifestPublisher:
    """Assemble and publish one immutable scientific run manifest.

    The publisher deliberately receives generic storage and metadata ports.  It
    does not create a mutable ``latest`` view and never puts local paths,
    staging IDs, run IDs, or timestamps in the canonical scientific payload.
    """

    def __init__(
        self,
        artifact_store: object,
        *,
        metadata_store: object | None = None,
        project_root: Path | str | None = None,
        environment_fingerprint: object | None = None,
        fingerprint_factory: Callable[..., object] | None = None,
        clock: object | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.metadata_store = metadata_store
        self.project_root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else Path.cwd().resolve()
        )
        self._fixed_environment_fingerprint = environment_fingerprint
        self._fingerprint_factory = fingerprint_factory or fingerprint_environment
        self.clock = clock

    def publish(
        self,
        result: object,
        config: object,
        *,
        environment_fingerprint: object | None = None,
    ) -> RunManifestPublication:
        """Build, checksum, publish, and index one successful run manifest."""

        evaluation = getattr(result, "evaluation", None)
        if evaluation is None:
            raise ValueError("successful run manifest requires evaluation output")
        evaluation_result = getattr(evaluation, "evaluation_result", None)
        if not isinstance(evaluation_result, EvaluationResult):
            raise TypeError("evaluation output must contain an EvaluationResult")

        configuration = _configuration_document(
            config,
            project_root=self.project_root,
        )
        configuration_checksum = _configuration_checksum(config)
        environment = self._environment_fingerprint(
            config,
            supplied=environment_fingerprint,
        )
        strategy_parameters = _strategy_parameters(config)
        snapshot_id = str(getattr(result, "snapshot_id", "")).strip()
        evaluation_range = getattr(result, "evaluation_range", None)
        if not snapshot_id:
            raise ValueError("run manifest requires snapshot_id")
        if evaluation_range is None:
            raise ValueError("run manifest requires evaluation_range")

        artifacts = _artifact_links(evaluation)
        scientific_artifacts = tuple(
            ScientificArtifactReference(item.role, item.checksum) for item in artifacts
        )
        content_identity = RunContentIdentity(
            schema_version=_RUN_MANIFEST_SCHEMA_VERSION,
            snapshot_id=snapshot_id,
            strategy_identifier=_strategy_identifier(config),
            strategy_parameters=strategy_parameters,
            evaluation_start=evaluation_range.start,
            evaluation_end=evaluation_range.end,
            configuration_checksum=configuration_checksum,
            environment_fingerprint=environment,
            scientific_artifacts=scientific_artifacts,
        )
        limitation = _limitation_disclosure(result, evaluation)
        run_manifest = RunManifest(
            content_identity=content_identity,
            operational_metadata=_operational_metadata(result, self._now()),
            limitation_disclosure=limitation,
            evaluation_result=evaluation_result,
        )
        # Constructing the domain object is intentional: it validates the
        # scientific/operational boundary before any bytes become durable.  The
        # published document below excludes its operational metadata so stable
        # reruns retain one scientific checksum.
        del run_manifest

        manifest = _manifest_document(
            result=result,
            evaluation=evaluation,
            content_identity=content_identity,
            configuration=configuration,
            environment=environment,
            strategy_parameters=strategy_parameters,
            limitation=limitation,
            artifacts=artifacts,
        )
        payload = canonical_json(manifest)
        manifest_checksum = sha256_bytes(payload)

        reference = self._publish_bytes(payload, manifest_checksum)
        self._record_evaluation_artifacts(artifacts)
        self._record_manifest(reference, manifest_checksum, len(payload))
        return RunManifestPublication(
            manifest=manifest,
            payload=payload,
            manifest_checksum=manifest_checksum,
            manifest_uri=reference.relative_uri,
            artifacts=artifacts,
            environment_fingerprint=environment,
        )

    publish_run_manifest = publish
    assemble_and_publish = publish

    def _environment_fingerprint(
        self,
        config: object,
        *,
        supplied: object | None,
    ) -> DomainEnvironmentFingerprint:
        seed = _deterministic_seed(config)
        raw = supplied
        if raw is None:
            raw = self._fixed_environment_fingerprint
        if raw is None:
            raw = self._fingerprint_factory(
                self.project_root,
                deterministic_seed=seed,
            )
        return _as_domain_environment_fingerprint(raw, seed=seed)

    def _publish_bytes(self, payload: bytes, checksum: str) -> ArtifactReference:
        create_staging = getattr(self.artifact_store, "create_staging", None)
        stage_bytes = getattr(self.artifact_store, "stage_bytes", None)
        publish_artifact = getattr(self.artifact_store, "publish_artifact", None)
        if (
            not callable(create_staging)
            or not callable(stage_bytes)
            or not callable(publish_artifact)
        ):
            raise TypeError(
                "run manifest publication requires create_staging, stage_bytes, "
                "and publish_artifact"
            )
        staging = create_staging(f"run-manifest-{checksum[:16]}-{uuid4().hex}")
        staged = stage_bytes(
            staging,
            f"run-manifest/{checksum}.json",
            payload,
            expected_checksum=checksum,
        )
        reference = publish_artifact(
            staged,
            metadata={
                "artifact_kind": "run_manifest",
                "checksum": checksum,
                "byte_size": len(payload),
                "media_type": _RUN_MANIFEST_MEDIA_TYPE,
                "schema_version": _RUN_MANIFEST_SCHEMA_VERSION,
                "row_count": None,
            },
        )
        relative_uri = getattr(reference, "relative_uri", None)
        reference_checksum = getattr(reference, "checksum", None)
        if not isinstance(relative_uri, str) or not relative_uri:
            raise ValueError("published run manifest has no relative URI")
        if reference_checksum != checksum:
            raise ValueError("published run manifest checksum did not match payload")
        return cast(ArtifactReference, reference)

    def _record_evaluation_artifacts(
        self,
        artifacts: Sequence[RunArtifactReference],
    ) -> None:
        if self.metadata_store is None:
            return
        record = getattr(self.metadata_store, "record_artifact", None)
        if not callable(record):
            raise TypeError("metadata store does not expose record_artifact")
        recorded: set[str] = set()
        created_at = self._now()
        for artifact in artifacts:
            if artifact.checksum in recorded:
                continue
            record(
                _object_reference(artifact),
                artifact_kind=artifact.role,
                created_at=created_at,
            )
            recorded.add(artifact.checksum)

    def _record_manifest(
        self,
        reference: ArtifactReference,
        checksum: str,
        byte_size: int,
    ) -> None:
        if self.metadata_store is None:
            return
        record = getattr(self.metadata_store, "record_artifact", None)
        if not callable(record):
            raise TypeError("metadata store does not expose record_artifact")
        record(
            ContentAddressedObjectRef(
                object_kind=ObjectKind.ARTIFACT,
                checksum=checksum,
                relative_uri=reference.relative_uri,
                schema_version=_RUN_MANIFEST_SCHEMA_VERSION,
                row_count=0,
                byte_size=byte_size,
                media_type=_RUN_MANIFEST_MEDIA_TYPE,
            ),
            artifact_kind="run_manifest",
            created_at=self._now(),
        )

    def _now(self) -> datetime:
        clock = self.clock
        if clock is None:
            value = datetime.now(UTC)
        elif callable(clock):
            value = clock()
        else:
            method = getattr(clock, "utc_now", None)
            if not callable(method):
                raise TypeError("manifest clock must expose utc_now")
            value = method()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("manifest clock must return an aware datetime")
        return value.astimezone(UTC)


def _configuration_document(
    config: object,
    *,
    project_root: Path | None = None,
) -> dict[str, object]:
    try:
        projected = non_secret_config(cast(ResolvedConfig, config))
        dumped = projected.model_dump(mode="json")
    except (TypeError, ValueError, AttributeError):
        if hasattr(config, "model_dump"):
            dumped = config.model_dump(mode="json")
        elif isinstance(config, Mapping):
            dumped = dict(config)
        else:
            dumped = {
                key: value
                for key, value in vars(config).items()
                if not key.startswith("_")
            }
    plain = _plain(dumped)
    if not isinstance(plain, dict):
        raise TypeError("non-secret configuration must serialize to a mapping")
    return _sanitize_configuration_paths(
        cast(dict[str, object], plain),
        project_root=project_root,
    )


_CONFIGURATION_PATH_FIELDS = frozenset(
    {
        "data_root",
        "artifact_root",
        "metadata_db",
        "mlflow_db",
        "local_secrets_file",
    }
)


def _sanitize_configuration_paths(
    document: dict[str, object],
    *,
    project_root: Path | None,
) -> dict[str, object]:
    """Remove machine-specific locations from the published config projection.

    Configuration resolution intentionally preserves normalized absolute paths
    for local I/O and for the established configuration checksum.  A run
    manifest is a scientific document, however, so those locations must not be
    copied into its canonical bytes.  Paths within the project remain useful as
    project-relative values; absolute paths outside the project collapse to a
    stable field-independent marker rather than leaking a machine location.
    """

    raw_paths = document.get("paths")
    if not isinstance(raw_paths, Mapping):
        return document

    root = project_root.resolve(strict=False) if project_root is not None else None
    sanitized_paths: dict[str, object] = {}
    for raw_name, value in raw_paths.items():
        name = str(raw_name)
        if name not in _CONFIGURATION_PATH_FIELDS:
            sanitized_paths[name] = value
            continue
        sanitized_paths[name] = _stable_local_path(value, root)

    sanitized = dict(document)
    sanitized["paths"] = sanitized_paths
    return sanitized


def _stable_local_path(value: object, project_root: Path | None) -> object:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        return value

    path = Path(value).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve(strict=False)
    if project_root is not None:
        try:
            return resolved.relative_to(project_root).as_posix() or "."
        except ValueError:
            pass
    return "<absolute-local-path>"


def _configuration_checksum(config: object) -> str:
    candidate = getattr(config, "configuration_checksum", None) or getattr(
        config, "non_secret_checksum", None
    )
    if isinstance(candidate, str) and len(candidate) == 64:
        return candidate
    try:
        if isinstance(config, ResolvedConfig):
            return sha256_bytes(ConfigurationSerializer().serialize(config))
        return sha256_canonical_json(_configuration_document(config))
    except Exception:
        return sha256_canonical_json({"configuration_type": type(config).__name__})


def _strategy_identifier(config: object) -> str:
    strategy = getattr(config, "strategy", config)
    value = getattr(strategy, "identifier", "monthly_momentum_v1")
    normalized = " ".join(str(value).split())
    if not normalized:
        raise ValueError("strategy identifier must not be blank")
    return normalized


def _strategy_parameters(config: object) -> MomentumStrategyParameters:
    strategy = getattr(config, "strategy", config)
    return MomentumStrategyParameters(
        position_count=int(getattr(strategy, "position_count", 1)),
        long_lookback_sessions=int(getattr(strategy, "long_lookback_sessions", 252)),
        skip_recent_sessions=int(getattr(strategy, "skip_recent_sessions", 21)),
    )


def _deterministic_seed(config: object) -> int:
    runtime = getattr(config, "runtime", config)
    value = getattr(runtime, "deterministic_seed", 0)
    if isinstance(value, bool):
        raise TypeError("deterministic_seed must be an integer")
    return int(value)


def _as_domain_environment_fingerprint(
    value: object,
    *,
    seed: int,
) -> DomainEnvironmentFingerprint:
    if isinstance(value, DomainEnvironmentFingerprint):
        return value
    if isinstance(value, InfrastructureEnvironmentFingerprint):
        distributions = value.installed_distributions
        return DomainEnvironmentFingerprint(
            python_version=value.python_version,
            operating_system=f"{value.os_name} {value.os_version}",
            architecture=value.architecture,
            dependencies=_normalized_dependencies(distributions),
            source_revision=value.source_revision or "unknown",
            source_dirty=value.source_dirty,
            deterministic_seed=value.deterministic_seed,
            effective_source_checksum=value.effective_source_checksum,
        )
    if isinstance(value, Mapping):
        nested = value.get("environment_fingerprint")
        if isinstance(nested, Mapping):
            return _as_domain_environment_fingerprint(nested, seed=seed)
        dependencies_value = value.get(
            "dependencies", value.get("installed_distributions", ())
        )
        dependencies: list[object] = []
        if isinstance(dependencies_value, Sequence) and not isinstance(
            dependencies_value, (str, bytes, bytearray)
        ):
            dependencies.extend(dependencies_value)
        normalized_dependencies = _normalized_dependencies(dependencies)
        operating_system = value.get("operating_system")
        if not isinstance(operating_system, str) or not operating_system.strip():
            operating_system = " ".join(
                str(part).strip()
                for part in (
                    value.get("os_name", "unknown"),
                    value.get("os_version", "unknown"),
                )
                if str(part).strip()
            )
        source_checksum = value.get("effective_source_checksum")
        if not isinstance(source_checksum, str) or len(source_checksum) != 64:
            source_checksum = sha256_canonical_json({"environment": _plain(value)})
        source_revision = value.get("source_revision") or "unknown"
        return DomainEnvironmentFingerprint(
            python_version=str(value.get("python_version", "unknown")),
            operating_system=str(operating_system),
            architecture=str(value.get("architecture", "unknown")),
            dependencies=normalized_dependencies,
            source_revision=str(source_revision),
            source_dirty=bool(value.get("source_dirty", False)),
            deterministic_seed=int(value.get("deterministic_seed", seed)),
            effective_source_checksum=source_checksum,
        )
    raise TypeError(
        "environment fingerprint must be the domain fingerprint, the local "
        "fingerprint, or a serializable mapping"
    )


def _dependency(name: object, version: object) -> DependencyVersion:
    return DependencyVersion(str(name), str(version))


def _normalized_dependencies(values: Sequence[object]) -> tuple[DependencyVersion, ...]:
    unique: dict[str, DependencyVersion] = {}
    for item in values:
        if isinstance(item, Mapping):
            name = item.get("name", "unknown")
            version = item.get("version", "unknown")
        elif isinstance(item, tuple) and len(item) == 2:
            name, version = item
        else:
            continue
        dependency = _dependency(name, version)
        unique[dependency.name] = dependency
    return tuple(unique[name] for name in sorted(unique))


def _operational_metadata(
    result: object, timestamp: datetime
) -> RunOperationalMetadata:
    run_id = getattr(result, "run_id", None)
    if not isinstance(run_id, UUID):
        run_id = UUID(str(run_id))
    return RunOperationalMetadata(
        run_id=run_id,
        state=RunState.SUCCEEDED,
        created_at=timestamp,
        started_at=timestamp,
        ended_at=timestamp,
    )


def _limitation_disclosure(
    result: object,
    evaluation: object,
) -> LimitationDisclosure:
    for candidate in (
        getattr(result, "limitation_disclosure", None),
        getattr(evaluation, "limitation_disclosure", None),
    ):
        if isinstance(candidate, LimitationDisclosure):
            return candidate
    return LimitationDisclosure.current()


def _artifact_links(evaluation: object) -> tuple[RunArtifactReference, ...]:
    candidates = getattr(evaluation, "artifacts", ())
    if isinstance(candidates, Mapping):
        candidates = tuple(candidates.values())
    links: list[RunArtifactReference] = []
    for candidate in candidates:
        checksum = getattr(candidate, "checksum", None)
        role = getattr(candidate, "role", None)
        reference = getattr(candidate, "reference", None)
        relative_uri = getattr(reference, "relative_uri", None)
        if relative_uri is None:
            relative_uri = getattr(candidate, "relative_uri", None)
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError("evaluation artifact must have a SHA-256 checksum")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("evaluation artifact must have a role")
        if not isinstance(relative_uri, str) or not relative_uri:
            raise ValueError(
                f"evaluation artifact {role!r} has not been published to CAS"
            )
        links.append(
            RunArtifactReference(
                checksum=checksum,
                role=role,
                relative_uri=relative_uri,
                byte_size=int(getattr(candidate, "byte_size", 0)),
                media_type=str(
                    getattr(candidate, "media_type", "application/octet-stream")
                ),
                schema_version=str(getattr(candidate, "schema_version", "artifact_v1")),
                row_count=(
                    int(cast(Any, candidate).row_count)
                    if getattr(candidate, "row_count", None) is not None
                    else None
                ),
            )
        )
    links.sort(key=lambda item: (item.role, item.checksum))
    keys = [(item.role, item.checksum) for item in links]
    if len(keys) != len(set(keys)):
        raise ValueError("evaluation artifacts must have unique role/checksum pairs")
    return tuple(links)


def _artifact_document(artifact: RunArtifactReference) -> dict[str, object]:
    return {
        "byte_size": artifact.byte_size,
        "checksum": artifact.checksum,
        "media_type": artifact.media_type,
        "role": artifact.role,
        "row_count": artifact.row_count,
        "schema_version": artifact.schema_version,
        "scientific": artifact.scientific,
    }


def _evaluation_document(
    evaluation: object,
    artifacts: Sequence[RunArtifactReference],
) -> object:
    serializer = getattr(evaluation, "to_serializable", None)
    value = serializer() if callable(serializer) else evaluation
    plain = _plain(value)
    if isinstance(plain, dict):
        document = _without_locations(plain)
        document["artifacts"] = [_artifact_document(item) for item in artifacts]
        return document
    return plain


def _core_document(result: object) -> object | None:
    core = getattr(result, "core_output", None)
    method = getattr(core, "to_scientific_dict", None)
    if callable(method):
        return _plain(method())
    return None


def _manifest_document(
    *,
    result: object,
    evaluation: object,
    content_identity: RunContentIdentity,
    configuration: Mapping[str, object],
    environment: DomainEnvironmentFingerprint,
    strategy_parameters: MomentumStrategyParameters,
    limitation: LimitationDisclosure,
    artifacts: Sequence[RunArtifactReference],
) -> dict[str, object]:
    evaluation_range = cast(Any, result).evaluation_range
    evaluation_result = cast(Any, evaluation).evaluation_result
    document: dict[str, object] = {
        "artifacts": [_artifact_document(item) for item in artifacts],
        "configuration": dict(configuration),
        "content_identity": content_identity.to_serializable(),
        "environment_fingerprint": environment.to_serializable(),
        "evaluation": _evaluation_document(evaluation, artifacts),
        "evaluation_end": evaluation_range.end,
        "evaluation_result": evaluation_result.to_serializable(),
        "evaluation_start": evaluation_range.start,
        "limitation_disclosure": {
            "lines": list(limitation.lines()),
            "version": limitation.version,
        },
        "snapshot_id": cast(Any, result).snapshot_id,
        "strategy_id": _strategy_identifier_from_identity(content_identity),
        "strategy_identifier": _strategy_identifier_from_identity(content_identity),
        "strategy_parameters": strategy_parameters.to_serializable(),
    }
    core = _core_document(result)
    if core is not None:
        document["core_output"] = core
    return document


def _strategy_identifier_from_identity(identity: RunContentIdentity) -> str:
    return identity.strategy_identifier


def _object_reference(artifact: RunArtifactReference) -> ContentAddressedObjectRef:
    return ContentAddressedObjectRef(
        object_kind=ObjectKind.ARTIFACT,
        checksum=artifact.checksum,
        relative_uri=artifact.relative_uri,
        schema_version=artifact.schema_version,
        row_count=artifact.row_count if artifact.row_count is not None else 0,
        byte_size=artifact.byte_size,
        media_type=artifact.media_type,
    )


def _without_locations(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    excluded = {"reference", "relative_uri", "uri", "path"}

    def visit(item: object, key: str = "") -> object:
        if isinstance(item, Mapping):
            return {
                str(name): visit(child, str(name))
                for name, child in item.items()
                if str(name) not in excluded
            }
        if isinstance(item, (tuple, list)):
            return [visit(child, key) for child in item]
        return item

    result = visit(value)
    return cast(dict[str, object], result)


def _plain(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    serializer = getattr(value, "to_serializable", None)
    if callable(serializer):
        return _plain(serializer())
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _plain(model_dump(mode="json"))
    return str(value)


__all__ = [
    "RunArtifactReference",
    "RunManifestPublication",
    "RunManifestPublisher",
]
