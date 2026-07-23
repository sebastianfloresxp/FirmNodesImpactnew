#!/usr/bin/env python3
"""Export precomputed semiconductor severity outputs for the static site.

The frontend expects:

  data/firms-index.json
  data/firms/{firm_id}.json

This script reads tabular precomputed firm-level outputs and writes those
files without introducing a server, database, or build framework.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".ndjson", ".parquet", ".feather"}

FATAL_REQUIRED_FIELDS = [
    "firm_id",
    "firm_name",
    "headline_score",
    "prime_pathway_loss",
    "network_fragmentation",
    "backup_capacity_loss",
]

PRODUCTION_FIELDS = [
    "country",
    "sector",
    "network_position",
    "rank",
    "total_firms_ranked",
    "affected_primes",
    "top_dependencies",
]

COLUMN_ALIASES = {
    "firm_id": ["firm_id", "id", "node_id", "company_id", "entity_id"],
    "firm_name": ["firm_name", "name", "company_name", "firm", "entity_name"],
    "aliases": ["aliases", "alias", "firm_aliases", "alternate_names"],
    "country": ["country", "jurisdiction", "country_or_jurisdiction"],
    "sector": ["sector", "category", "industry", "firm_type", "value_chain_role"],
    "network_position": ["network_position", "position", "graph_position", "node_position"],
    "headline_score": [
        "headline_score",
        "severity_score",
        "structural_severity_score",
        "structural_severity",
        "score",
    ],
    "severity_band": ["severity_band", "band"],
    "prime_pathway_loss": [
        "prime_pathway_loss",
        "dod_prime_pathway_loss",
        "pathway_loss",
        "h1_loss",
    ],
    "network_fragmentation": [
        "network_fragmentation",
        "fragmentation",
        "fragmentation_delta",
        "h2_fragmentation",
    ],
    "backup_capacity_loss": [
        "backup_capacity_loss",
        "redundancy_loss",
        "alternate_pathway_loss",
        "backup_loss",
    ],
    "rank": ["rank", "severity_rank", "structural_rank"],
    "total_firms_ranked": ["total_firms_ranked", "total_ranked", "firm_count", "n_firms"],
    "affected_primes": ["affected_primes", "top_affected_primes", "prime_impacts"],
    "top_dependencies": ["top_dependencies", "dependencies", "dependency_drivers"],
    "geographic_exposure": ["geographic_exposure", "geo_exposure", "country_exposure"],
    "network_role": ["network_role", "role", "role_label"],
    "severity_drivers": ["severity_drivers", "drivers", "driver_labels"],
    "updated_at": ["updated_at", "exported_at"],
}

INDEX_FIELDS = [
    "id",
    "name",
    "aliases",
    "country",
    "sector",
    "network_position",
    "headline_score",
    "severity_band",
]

METRIC_DESCRIPTIONS = {
    "prime_pathway_loss": "Share of mapped DoD primes losing one or more semiconductor input pathways after the selected firm is removed.",
    "network_fragmentation": "Increase in fragmentation across the reachable supply network after removal.",
    "backup_capacity_loss": "Reduction in alternate pathways available under fixed-substitution assumptions.",
}


class ValidationIssue:
    def __init__(self, level: str, row: Optional[int], field: str, message: str) -> None:
        self.level = level
        self.row = row
        self.field = field
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "row": self.row,
            "field": self.field,
            "message": self.message,
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.output_root).resolve()
    data_dir = repo_root / args.data_dir
    firms_dir = data_dir / "firms"
    reports_dir = data_dir / "reports"

    records = load_records(Path(args.input))
    if args.limit is not None:
        records = records[: args.limit]

    normalized_rows, issues = normalize_records(records)
    validation_report = validate_rows(normalized_rows, issues, strict=args.strict)
    write_validation_reports(validation_report, reports_dir)

    has_errors = validation_report["summary"]["errors"] > 0
    if has_errors and not args.force:
        print("Validation failed. Reports written to:", reports_dir, file=sys.stderr)
        print("Use --force to export despite validation errors.", file=sys.stderr)
        return 1

    if args.clean and firms_dir.exists():
        shutil.rmtree(firms_dir)
    firms_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(args, len(normalized_rows))
    scenarios, index_records = build_site_records(normalized_rows, metadata)

    scenario_files = write_scenarios(scenarios, firms_dir, pretty=args.pretty)
    index_path = data_dir / "firms-index.json"
    write_json(index_path, index_records, pretty=args.pretty)
    write_json(data_dir / "firms-index.min.json", index_records, pretty=False)
    write_gzip(index_path, data_dir / "firms-index.json.gz")
    write_json(data_dir / "metadata.json", metadata, pretty=True)

    stats = build_export_stats(
        data_dir=data_dir,
        index_path=index_path,
        scenario_files=scenario_files,
        firms_exported=len(index_records),
    )
    write_json(reports_dir / "export-stats.json", stats, pretty=True)
    print_stats(stats, reports_dir)
    return 0


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export precomputed firm-level severity outputs for the static GitHub Pages site."
    )
    parser.add_argument("input", help="Input file or directory: csv, json, jsonl, ndjson, parquet, or feather.")
    parser.add_argument("--output-root", default=".", help="Repository root containing index.html. Default: current directory.")
    parser.add_argument("--data-dir", default="data", help="Data directory relative to output root. Default: data.")
    parser.add_argument("--metadata", help="Optional metadata JSON file to merge into data/metadata.json.")
    parser.add_argument("--snapshot-date", default="June 2025", help="Network snapshot label.")
    parser.add_argument("--firm-count", type=int, default=78000, help="Firm count shown in metadata.")
    parser.add_argument("--relationship-count", type=int, default=340000, help="Supplier relationship count shown in metadata.")
    parser.add_argument("--project-title", default="Structural Severity Explorer", help="Project title in metadata.")
    parser.add_argument("--clean", action="store_true", help="Remove existing data/firms/*.json before writing scenarios.")
    parser.add_argument("--force", action="store_true", help="Write output even when validation has errors.")
    parser.add_argument("--strict", action="store_true", help="Treat production completeness warnings as validation errors.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print scenario and index JSON. Default is compact JSON.")
    parser.add_argument("--limit", type=int, help="Export only the first N rows. Useful for smoke tests.")
    return parser.parse_args(argv)


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Input path does not exist: {path}")

    files = expand_input_files(path)
    records: List[Dict[str, Any]] = []
    for file_path in files:
        records.extend(read_file_records(file_path))
    return records


def expand_input_files(path: Path) -> List[Path]:
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise SystemExit(f"Unsupported input file type: {path}")
        return [path]

    files = [
        child
        for child in sorted(path.iterdir())
        if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files:
        raise SystemExit(f"No supported input files found in: {path}")
    return files


def read_file_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return ensure_record_list(rows, path)

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return json_payload_to_records(payload, path)

    if suffix in {".parquet", ".feather"}:
        return read_tabular_with_pandas(path)

    raise SystemExit(f"Unsupported input file type: {path}")


def json_payload_to_records(payload: Any, path: Path) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return ensure_record_list(payload, path)
    if isinstance(payload, dict):
        for key in ("records", "data", "firms", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return ensure_record_list(value, path)
    raise SystemExit(f"JSON input must be an array or contain records/data/firms/results: {path}")


def ensure_record_list(rows: Iterable[Any], path: Path) -> List[Dict[str, Any]]:
    records = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise SystemExit(f"Row {index} in {path} is not an object.")
        records.append(dict(row))
    return records


def read_tabular_with_pandas(path: Path) -> List[Dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Reading Parquet or Feather requires pandas and an installed engine such as pyarrow. "
            "Install them or export the source data as CSV/JSON."
        ) from exc

    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_feather(path)
    frame = frame.where(frame.notna(), None)
    return frame.to_dict(orient="records")


def normalize_records(records: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[ValidationIssue]]:
    normalized_rows = []
    issues: List[ValidationIssue] = []
    for row_number, row in enumerate(records, start=1):
        normalized = normalize_row(row, row_number, issues)
        normalized_rows.append(normalized)
    return normalized_rows, issues


def normalize_row(row: Mapping[str, Any], row_number: int, issues: List[ValidationIssue]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {"_row_number": row_number}
    for canonical in COLUMN_ALIASES:
        raw_value = get_first_present(row, COLUMN_ALIASES[canonical])
        normalized[canonical] = coerce_field(canonical, raw_value, row_number, issues)

    firm_id = normalized.get("firm_id")
    if firm_id is not None:
        normalized["firm_id"] = str(firm_id).strip()

    firm_name = normalized.get("firm_name")
    if firm_name is not None:
        normalized["firm_name"] = str(firm_name).strip()

    return normalized


def get_first_present(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    lowered = {str(key).lower(): key for key in row.keys()}
    for alias in aliases:
        key = lowered.get(alias.lower())
        if key is not None:
            return row[key]
    return None


def coerce_field(field: str, value: Any, row_number: int, issues: List[ValidationIssue]) -> Any:
    if is_missing(value):
        if field in {"aliases", "affected_primes", "top_dependencies", "geographic_exposure", "severity_drivers"}:
            return []
        return None

    if field in {
        "headline_score",
        "prime_pathway_loss",
        "network_fragmentation",
        "backup_capacity_loss",
        "rank",
        "total_firms_ranked",
    }:
        return coerce_number(field, value, row_number, issues)

    if field in {"aliases", "affected_primes", "top_dependencies", "geographic_exposure", "severity_drivers"}:
        return coerce_list(value)

    if isinstance(value, (dict, list)):
        return value
    return str(value).strip()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def coerce_number(field: str, value: Any, row_number: int, issues: List[ValidationIssue]) -> Optional[float]:
    if is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    else:
        text = str(value).strip().replace(",", "")
        is_percent = text.endswith("%")
        if is_percent:
            text = text[:-1].strip()
        try:
            number = float(text)
        except ValueError:
            issues.append(ValidationIssue("error", row_number, field, f"Expected numeric value, found {value!r}."))
            return None
        if is_percent and field != "headline_score":
            number = number / 100.0
    if not math.isfinite(number):
        issues.append(ValidationIssue("error", row_number, field, f"Numeric value is not finite: {value!r}."))
        return None
    if field in {"rank", "total_firms_ranked"}:
        return int(number)
    return number


def coerce_list(value: Any) -> List[Any]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        return [clean_json_value(item) for item in value if not is_missing(item)]
    if isinstance(value, tuple):
        return [clean_json_value(item) for item in value if not is_missing(item)]
    if isinstance(value, dict):
        return [clean_json_value(value)]

    text = str(value).strip()
    if not text:
        return []
    parsed = parse_structured_text(text)
    if isinstance(parsed, list):
        return [clean_json_value(item) for item in parsed if not is_missing(item)]
    if isinstance(parsed, dict):
        return [clean_json_value(parsed)]
    if parsed is not None and not isinstance(parsed, str):
        return [clean_json_value(parsed)]

    separator = ";" if ";" in text else ","
    return [part.strip() for part in text.split(separator) if part.strip()]


def parse_structured_text(text: str) -> Any:
    if not text:
        return None
    if text[0] in "[{\"'":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return None
    return None


def clean_json_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean_json_value(item) for key, item in value.items() if not is_missing(item)}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value if not is_missing(item)]
    if isinstance(value, tuple):
        return [clean_json_value(item) for item in value if not is_missing(item)]
    return value


def validate_rows(rows: Sequence[Dict[str, Any]], issues: List[ValidationIssue], strict: bool) -> Dict[str, Any]:
    ids = [row.get("firm_id") for row in rows if row.get("firm_id")]
    duplicates = {firm_id for firm_id, count in Counter(ids).items() if count > 1}
    field_coverage: Dict[str, int] = Counter()

    for row in rows:
        row_number = int(row["_row_number"])
        for field in FATAL_REQUIRED_FIELDS:
            if is_missing(row.get(field)):
                issues.append(ValidationIssue("error", row_number, field, "Required field is missing."))

        for field in PRODUCTION_FIELDS:
            if is_missing(row.get(field)):
                level = "error" if strict else "warning"
                issues.append(ValidationIssue(level, row_number, field, "Production field is missing."))

        firm_id = row.get("firm_id")
        if firm_id:
            if firm_id in duplicates:
                issues.append(ValidationIssue("error", row_number, "firm_id", f"Duplicate firm_id: {firm_id}."))
            if not is_safe_firm_id(str(firm_id)):
                issues.append(
                    ValidationIssue(
                        "error",
                        row_number,
                        "firm_id",
                        "firm_id must be URL and file safe: letters, numbers, period, underscore, or hyphen.",
                    )
                )

        validate_numeric_ranges(row, row_number, issues)
        for field, value in row.items():
            if field.startswith("_"):
                continue
            if not is_missing(value):
                field_coverage[field] += 1

    issue_counts = Counter(issue.level for issue in issues)
    return {
        "summary": {
            "rows_read": len(rows),
            "errors": issue_counts.get("error", 0),
            "warnings": issue_counts.get("warning", 0),
            "duplicate_ids": len(duplicates),
        },
        "field_coverage": dict(sorted(field_coverage.items())),
        "duplicate_ids": sorted(duplicates),
        "issues": [issue.to_dict() for issue in issues],
    }


def validate_numeric_ranges(row: Mapping[str, Any], row_number: int, issues: List[ValidationIssue]) -> None:
    score = row.get("headline_score")
    if score is not None and not (0 <= float(score) <= 100):
        issues.append(ValidationIssue("error", row_number, "headline_score", "headline_score must be between 0 and 100."))

    for field in ("prime_pathway_loss", "network_fragmentation", "backup_capacity_loss"):
        value = row.get(field)
        if value is None:
            continue
        numeric = float(value)
        if numeric < 0 or numeric > 100:
            issues.append(ValidationIssue("error", row_number, field, f"{field} must be between 0 and 100."))
        elif field != "network_fragmentation" and numeric > 1:
            issues.append(
                ValidationIssue(
                    "warning",
                    row_number,
                    field,
                    f"{field} is greater than 1 and will be treated as a 0-100 percentage scale by the frontend.",
                )
            )

    rank = row.get("rank")
    if rank is not None and int(rank) < 1:
        issues.append(ValidationIssue("error", row_number, "rank", "rank must be 1 or greater."))

    total = row.get("total_firms_ranked")
    if total is not None and int(total) < 1:
        issues.append(ValidationIssue("error", row_number, "total_firms_ranked", "total_firms_ranked must be 1 or greater."))

    if rank is not None and total is not None and int(rank) > int(total):
        issues.append(ValidationIssue("warning", row_number, "rank", "rank is greater than total_firms_ranked."))


def is_safe_firm_id(firm_id: str) -> bool:
    if firm_id in {"", ".", ".."}:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return all(char in allowed for char in firm_id)


def write_validation_reports(report: Mapping[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "validation-report.json", report, pretty=True)
    write_validation_markdown(report, reports_dir / "validation-report.md")


def write_validation_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Static Export Validation Report",
        "",
        f"- Rows read: {summary['rows_read']}",
        f"- Errors: {summary['errors']}",
        f"- Warnings: {summary['warnings']}",
        f"- Duplicate IDs: {summary['duplicate_ids']}",
        "",
        "## Field Coverage",
        "",
    ]
    for field, count in report["field_coverage"].items():
        lines.append(f"- `{field}`: {count}")

    lines.extend(["", "## Issues", ""])
    issues = report.get("issues", [])
    if not issues:
        lines.append("No validation issues reported.")
    else:
        for issue in issues[:1000]:
            row = issue["row"] if issue["row"] is not None else "global"
            lines.append(f"- `{issue['level']}` row `{row}` field `{issue['field']}`: {issue['message']}")
        if len(issues) > 1000:
            lines.append(f"- Truncated: {len(issues) - 1000} additional issues in validation-report.json.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_metadata(args: argparse.Namespace, row_count: int) -> Dict[str, Any]:
    metadata = {
        "project_title": args.project_title,
        "network_snapshot": args.snapshot_date,
        "firm_count": args.firm_count or row_count,
        "relationship_count": args.relationship_count,
        "methodology_label": "Single-firm removal under fixed substitution",
        "disclaimer": "For research and policy analysis. Measures structural consequence, not probability.",
        "sample_data": False,
        "exported_at": date.today().isoformat(),
    }
    if args.metadata:
        metadata_path = Path(args.metadata)
        with metadata_path.open("r", encoding="utf-8") as handle:
            user_metadata = json.load(handle)
        if not isinstance(user_metadata, dict):
            raise SystemExit("--metadata must point to a JSON object.")
        metadata.update(user_metadata)
    return metadata


def build_site_records(rows: Sequence[Dict[str, Any]], metadata: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered_rows = sorted(rows, key=sort_key_for_row)
    scenarios: List[Dict[str, Any]] = []
    index_records: List[Dict[str, Any]] = []
    for row in ordered_rows:
        scenario = build_scenario(row, metadata)
        scenarios.append(scenario)
        index_records.append({field: scenario[field] for field in INDEX_FIELDS if field in scenario and not is_empty_index_value(scenario[field])})
    return scenarios, index_records


def sort_key_for_row(row: Mapping[str, Any]) -> Tuple[int, float, str]:
    rank = row.get("rank")
    if rank is not None:
        return (0, float(rank), str(row.get("firm_name") or ""))
    score = row.get("headline_score")
    return (1, -(float(score) if score is not None else -1), str(row.get("firm_name") or ""))


def build_scenario(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> Dict[str, Any]:
    firm_id = str(row["firm_id"])
    name = str(row["firm_name"])
    score = round_float(row.get("headline_score"))
    scenario: Dict[str, Any] = {
        "id": firm_id,
        "name": name,
        "aliases": normalize_aliases(row.get("aliases")),
        "country": row.get("country") or "",
        "sector": row.get("sector") or "",
        "network_position": row.get("network_position") or "",
        "headline_score": score,
        "severity_band": row.get("severity_band") or get_severity_band(score),
        "rank": int(row["rank"]) if row.get("rank") is not None else None,
        "total_firms_ranked": int(row["total_firms_ranked"]) if row.get("total_firms_ranked") is not None else metadata.get("firm_count"),
        "component_metrics": {
            "prime_pathway_loss": build_metric("prime_pathway_loss", row.get("prime_pathway_loss")),
            "network_fragmentation": build_metric("network_fragmentation", row.get("network_fragmentation")),
            "backup_capacity_loss": build_metric("backup_capacity_loss", row.get("backup_capacity_loss")),
        },
        "affected_primes": normalize_named_list(row.get("affected_primes")),
        "methodology": {
            "snapshot_date": metadata.get("network_snapshot", "June 2025"),
            "assumption": "Fixed substitution",
            "what_it_measures": "Consequence of removing this firm from the directed supply network.",
            "what_it_does_not_measure": "Probability of disruption, likelihood of attack, firm intent, financial risk, or classified dependency.",
            "notes": "Scores are based on precomputed single-firm removal scenarios.",
        },
        "updated_at": row.get("updated_at") or metadata.get("exported_at") or date.today().isoformat(),
    }

    optional_fields = {
        "top_dependencies": normalize_named_list(row.get("top_dependencies")),
        "geographic_exposure": normalize_named_list(row.get("geographic_exposure")),
        "network_role": row.get("network_role") or "",
        "severity_drivers": normalize_named_list(row.get("severity_drivers")),
    }
    for field, value in optional_fields.items():
        if not is_missing(value) and value != []:
            scenario[field] = value

    return {key: value for key, value in scenario.items() if value is not None}


def normalize_aliases(value: Any) -> List[str]:
    aliases = normalize_named_list(value)
    return [item for item in aliases if isinstance(item, str)]


def normalize_named_list(value: Any) -> List[Any]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        return [clean_json_value(item) for item in value if not is_missing(item)]
    return coerce_list(value)


def build_metric(key: str, value: Any) -> Dict[str, Any]:
    numeric = round_float(value)
    return {
        "value": numeric,
        "display": format_metric_display(key, numeric),
        "description": METRIC_DESCRIPTIONS[key],
    }


def format_metric_display(key: str, value: Optional[float]) -> str:
    if value is None:
        return "Not reported"
    if key == "network_fragmentation":
        return f"{value:.2f}"
    percent = value * 100 if value <= 1 else value
    return f"{percent:.0f}%"


def round_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return round(number, 6)


def get_severity_band(score: Optional[float]) -> str:
    if score is None:
        return "Unreported"
    if score < 20:
        return "Low"
    if score < 40:
        return "Moderate"
    if score < 60:
        return "Elevated"
    if score < 80:
        return "High"
    return "Critical"


def is_empty_index_value(value: Any) -> bool:
    return value is None or value == "" or value == []


def write_scenarios(scenarios: Sequence[Mapping[str, Any]], firms_dir: Path, pretty: bool) -> List[Path]:
    written = []
    for scenario in scenarios:
        path = firms_dir / f"{scenario['id']}.json"
        write_json(path, scenario, pretty=pretty)
        written.append(path)
    return written


def write_json(path: Path, payload: Any, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if pretty:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=False)
        else:
            json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"), sort_keys=False)
        handle.write("\n")


def write_gzip(source: Path, destination: Path) -> None:
    with source.open("rb") as input_handle, gzip.open(destination, "wb", compresslevel=9) as output_handle:
        shutil.copyfileobj(input_handle, output_handle)


def build_export_stats(data_dir: Path, index_path: Path, scenario_files: Sequence[Path], firms_exported: int) -> Dict[str, Any]:
    file_sizes = {str(path.relative_to(data_dir.parent)): path.stat().st_size for path in scenario_files}
    largest_path = max(scenario_files, key=lambda path: path.stat().st_size) if scenario_files else None
    total_output_size = sum(path.stat().st_size for path in data_dir.rglob("*") if path.is_file())
    return {
        "firms_exported": firms_exported,
        "scenario_files_written": len(scenario_files),
        "index_size_bytes": index_path.stat().st_size if index_path.exists() else 0,
        "index_gzip_size_bytes": (data_dir / "firms-index.json.gz").stat().st_size if (data_dir / "firms-index.json.gz").exists() else 0,
        "total_output_size_bytes": total_output_size,
        "largest_scenario_file": {
            "path": str(largest_path.relative_to(data_dir.parent)) if largest_path else None,
            "size_bytes": largest_path.stat().st_size if largest_path else 0,
        },
        "scenario_file_size_bytes": file_sizes,
    }


def print_stats(stats: Mapping[str, Any], reports_dir: Path) -> None:
    largest = stats["largest_scenario_file"]
    print("Static export complete")
    print(f"  Firms exported: {stats['firms_exported']}")
    print(f"  Scenario files written: {stats['scenario_files_written']}")
    print(f"  Index size: {format_bytes(stats['index_size_bytes'])}")
    print(f"  Index gzip size: {format_bytes(stats['index_gzip_size_bytes'])}")
    print(f"  Total output size: {format_bytes(stats['total_output_size_bytes'])}")
    print(f"  Largest scenario file: {largest['path']} ({format_bytes(largest['size_bytes'])})")
    print(f"  Reports: {reports_dir}")


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


if __name__ == "__main__":
    raise SystemExit(main())
