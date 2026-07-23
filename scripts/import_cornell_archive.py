#!/usr/bin/env python3
"""Build static site data from the provided Cornell semiconductor archive.

The supplied archive contains report-ready Chapter 4 tables and local
case-study CSVs, not the full 78k-row per-firm removal export. This importer
uses the real available outputs:

- report/tables/tab_4.2_strict_top25_observed.tex
- report/tables/tab_c.2_adjacent_uncertain_top25_empirical.tex
- figs/chapter4/v2_fix01/main_text/fig_4_4_case_study_network_nodes.csv
- figs/chapter4/v2_fix01/main_text/fig_4_4_case_study_network_edges.csv

It writes the same data contract consumed by the existing frontend.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


STRICT_TABLE = "cornell-semiconductor-project-main/report/tables/tab_4.2_strict_top25_observed.tex"
ADJACENT_TABLE = "cornell-semiconductor-project-main/report/tables/tab_c.2_adjacent_uncertain_top25_empirical.tex"
CASE_NODES = "cornell-semiconductor-project-main/figs/chapter4/v2_fix01/main_text/fig_4_4_case_study_network_nodes.csv"
CASE_EDGES = "cornell-semiconductor-project-main/figs/chapter4/v2_fix01/main_text/fig_4_4_case_study_network_edges.csv"

NETWORK_NODES = 78257
NETWORK_EDGES = 340445
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1]

LEGAL_SUFFIXES = {
    "ag",
    "asa",
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "group",
    "holding",
    "holdings",
    "inc",
    "incorporated",
    "international",
    "ltd",
    "llc",
    "lp",
    "plc",
    "sa",
    "sas",
    "se",
    "spa",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    archive = Path(args.archive).resolve()
    output_root = Path(args.output_root).resolve()
    data_dir = output_root / "data"
    firms_dir = data_dir / "firms"
    reports_dir = data_dir / "reports"

    if not archive.exists():
        raise SystemExit(f"Archive does not exist: {archive}")

    with zipfile.ZipFile(archive) as zf:
        strict_rows = parse_top25_table(read_text(zf, STRICT_TABLE), "strict")
        adjacent_rows = parse_top25_table(read_text(zf, ADJACENT_TABLE), "adjacent")
        nodes = read_case_nodes(zf)
        edges = read_case_edges(zf)

    all_rows = strict_rows + adjacent_rows
    scenarios, index_records = build_site_records(all_rows, nodes, edges)

    if args.clean and firms_dir.exists():
        shutil.rmtree(firms_dir)
    firms_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        write_json(firms_dir / f"{scenario['id']}.json", scenario, pretty=args.pretty)

    write_json(data_dir / "firms-index.json", index_records, pretty=args.pretty)
    write_json(data_dir / "firms-index.min.json", index_records, pretty=False)
    write_gzip(data_dir / "firms-index.json", data_dir / "firms-index.json.gz")
    write_json(data_dir / "metadata.json", build_metadata(len(index_records)), pretty=True)
    write_json(reports_dir / "cornell-archive-source-coverage.json", build_source_coverage(scenarios), pretty=True)
    write_json(reports_dir / "export-stats.json", build_export_stats(data_dir, firms_dir, len(index_records)), pretty=True)

    print(f"Exported {len(index_records)} firm scenarios from supplied archive.")
    print(f"Output: {data_dir}")
    print("Note: the archive contains Top-25 report tables, not the full 78k-row scenario export.")
    return 0


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export site data from the provided Cornell semiconductor archive.")
    parser.add_argument(
        "archive",
        nargs="?",
        default="/Users/sebastianflores/Downloads/cornell-semiconductor-project-main.zip",
        help="Path to cornell-semiconductor-project-main.zip.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Repository root for the static site.")
    parser.add_argument("--clean", action="store_true", help="Remove existing data/firms before writing.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print generated JSON.")
    return parser.parse_args(argv)


def read_text(zf: zipfile.ZipFile, member: str) -> str:
    with zf.open(member) as handle:
        return handle.read().decode("utf-8")


def parse_top25_table(text: str, stream: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    max_h1 = 0.0
    for line in text.splitlines():
        cleaned = line.strip()
        if "&" not in cleaned or cleaned.startswith("%") or cleaned.startswith("\\"):
            continue
        cleaned = cleaned.rstrip("\\").strip()
        cleaned = cleaned.replace("\\&", "__AMPERSAND__")
        parts = [clean_latex(part.replace("__AMPERSAND__", "\\&")) for part in cleaned.split("&")]
        if len(parts) != 5 or not parts[0].strip().isdigit():
            continue
        rank = int(parts[0])
        h1_log = float(parts[4])
        max_h1 = max(max_h1, h1_log)
        if stream == "strict":
            sector = parts[2]
            network_role = "Strict value-chain chokepoint"
            source_table = "Table 4.2 strict observed Top-25 value-chain chokepoints"
        else:
            sector = f"Adjacent/uncertain candidate ({parts[2]})"
            network_role = "Adjacent/uncertain validation candidate"
            source_table = "Table C.2 empirical Top-25 adjacent/uncertain candidates"

        rows.append(
            {
                "rank": rank,
                "firm": parts[1],
                "sector": sector,
                "table_role": parts[2],
                "network_position": parts[3],
                "h1_log": h1_log,
                "stream": stream,
                "network_role": network_role,
                "source_table": source_table,
            }
        )

    for row in rows:
        row["headline_score"] = round((row["h1_log"] / max_h1) * 100, 1) if max_h1 else None
    return rows


def clean_latex(text: str) -> str:
    text = text.strip()
    text = text.replace("\\&", "&")
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_case_nodes(zf: zipfile.ZipFile) -> Dict[str, Dict[str, Any]]:
    with zf.open(CASE_NODES) as handle:
        reader = csv.DictReader(line.decode("utf-8") for line in handle)
        nodes = {}
        for row in reader:
            node = dict(row)
            node["is_selected_prime"] = parse_bool(node.get("is_selected_prime"))
            node["is_prime_degraded_after_removal"] = parse_bool(node.get("is_prime_degraded_after_removal"))
            node["is_prime_disconnected_after_removal"] = parse_bool(node.get("is_prime_disconnected_after_removal"))
            node["is_top25_strict"] = parse_bool(node.get("is_top25_strict"))
            node["is_focal"] = parse_bool(node.get("is_focal"))
            nodes[node["analysis_uid"]] = node
        return nodes


def read_case_edges(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    with zf.open(CASE_EDGES) as handle:
        reader = csv.DictReader(line.decode("utf-8") for line in handle)
        return [(row["src_uid"], row["dst_uid"]) for row in reader]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_site_records(
    rows: Sequence[Mapping[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    name_to_node = build_name_index(nodes)
    dependencies = build_dependency_index(nodes, edges)
    degraded_primes = [
        build_prime_record(node, "Degraded")
        for node in nodes.values()
        if node.get("is_prime_degraded_after_removal")
    ]

    scenarios = []
    for row in rows:
        node = find_node_for_firm(row["firm"], name_to_node)
        firm_id = node_id_to_firm_id(node["analysis_uid"]) if node else make_slug_id(row["stream"], row["firm"])
        h1_label = f"H1_log {row['h1_log']:.4f}"
        is_strict = row["stream"] == "strict"

        affected_primes = []
        if node and node.get("is_focal"):
            affected_primes = degraded_primes
        elif node:
            affected_primes = dependencies.get(node["analysis_uid"], {}).get("neighbor_primes", [])[:10]

        top_dependencies = dependencies.get(node["analysis_uid"], {}).get("dependencies", [])[:12] if node else []
        scenario = {
            "id": firm_id,
            "name": row["firm"],
            "aliases": build_aliases(row["firm"], node),
            "country": "Not reported in supplied table",
            "sector": row["sector"],
            "network_position": row["network_position"],
            "network_role": row["network_role"],
            "headline_score": row["headline_score"],
            "severity_band": get_severity_band(row["headline_score"]),
            "rank": row["rank"],
            "total_firms_ranked": 25,
            "component_metrics": {
                "prime_pathway_loss": {
                    "value": row["h1_log"],
                    "display": h1_label,
                    "description": "Reported log-obligation-weighted deny severity loss from the supplied Chapter 4 table.",
                },
                "network_fragmentation": {
                    "value": None,
                    "display": "Not reported",
                    "description": "The supplied archive table does not include firm-level fragmentation for this row.",
                },
                "backup_capacity_loss": {
                    "value": None,
                    "display": "Not reported",
                    "description": "The supplied archive table does not include firm-level backup-capacity loss for this row.",
                },
            },
            "affected_primes": affected_primes,
            "top_dependencies": top_dependencies,
            "geographic_exposure": [],
            "severity_drivers": [
                row["source_table"],
                f"Reported {h1_label}",
                "Empirical observed view",
                "Strict semiconductor relevance lens" if is_strict else "Validation queue; semiconductor relevance unresolved",
            ],
            "methodology": {
                "snapshot_date": "June 2025",
                "assumption": "Fixed substitution",
                "what_it_measures": "Consequence of removing this firm from the directed supply network.",
                "what_it_does_not_measure": "Probability of disruption, likelihood of attack, firm intent, financial risk, or classified dependency.",
                "notes": "Generated from the supplied dissertation archive. Headline scores are normalized within each supplied Top-25 table because the full 78k-row firm-level scenario export was not included.",
            },
            "updated_at": date.today().isoformat(),
            "source": {
                "archive": "cornell-semiconductor-project-main.zip",
                "table": row["source_table"],
                "raw_h1_log": row["h1_log"],
                "source_stream": row["stream"],
                "case_study_node_id": node["analysis_uid"] if node else None,
            },
        }
        scenarios.append(scenario)

    index_records = []
    for scenario in scenarios:
        index_records.append(
            {
                "id": scenario["id"],
                "name": scenario["name"],
                "aliases": scenario["aliases"],
                "country": scenario["country"],
                "sector": scenario["sector"],
                "network_position": scenario["network_position"],
                "headline_score": scenario["headline_score"],
                "severity_band": scenario["severity_band"],
            }
        )
    return scenarios, index_records


def build_name_index(nodes: Mapping[str, Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    index = {}
    for node in nodes.values():
        name = str(node.get("name", ""))
        for key in {normalize_name(name), simplify_name(name)}:
            if key:
                index[key] = node
    return index


def find_node_for_firm(name: str, index: Mapping[str, Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for key in (normalize_name(name), simplify_name(name)):
        if key in index:
            return index[key]

    simple = simplify_name(name)
    for key, node in index.items():
        if simple and (key.startswith(simple) or simple.startswith(key)):
            return node
    return None


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def simplify_name(name: str) -> str:
    tokens = [token for token in normalize_name(name).split() if token not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def build_dependency_index(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Tuple[str, str]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    output: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: {"dependencies": [], "neighbor_primes": []})
    for src, dst in edges:
        src_node = nodes.get(src)
        dst_node = nodes.get(dst)
        if not src_node or not dst_node:
            continue

        output[src]["dependencies"].append(build_dependency_record(dst_node, "Outgoing local tie"))
        output[dst]["dependencies"].append(build_dependency_record(src_node, "Incoming local tie"))

        if is_prime_node(dst_node):
            output[src]["neighbor_primes"].append(build_prime_record(dst_node, "Local neighbor"))
        if is_prime_node(src_node):
            output[dst]["neighbor_primes"].append(build_prime_record(src_node, "Local neighbor"))
    return output


def build_dependency_record(node: Mapping[str, Any], relation: str) -> Dict[str, Any]:
    return {
        "name": node.get("name", "Unnamed node"),
        "role": node.get("role", "Not reported"),
        "relation": relation,
    }


def build_prime_record(node: Mapping[str, Any], impact_level: str) -> Dict[str, Any]:
    return {
        "name": node.get("name", "Unnamed prime"),
        "impact_level": impact_level,
        "pathways_lost": "Not reported",
        "remaining_pathways": "Not reported",
    }


def is_prime_node(node: Mapping[str, Any]) -> bool:
    return "Prime Endpoint" in str(node.get("role", "")) or bool(node.get("is_selected_prime"))


def build_aliases(firm_name: str, node: Optional[Mapping[str, Any]]) -> List[str]:
    aliases = []
    if node and node.get("name") and node["name"] != firm_name:
        aliases.append(str(node["name"]))
    simple = simplify_name(firm_name)
    if simple and simple != normalize_name(firm_name):
        aliases.append(simple.title())
    return sorted(set(aliases))


def node_id_to_firm_id(node_id: str) -> str:
    return node_id.replace(":", "-")


def make_slug_id(stream: str, firm_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", firm_name.lower()).strip("-")
    return f"{stream}-{slug}"[:96]


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


def build_metadata(firms_exported: int) -> Dict[str, Any]:
    return {
        "project_title": "Structural Severity Explorer",
        "network_snapshot": "June 2025",
        "firm_count": NETWORK_NODES,
        "relationship_count": NETWORK_EDGES,
        "methodology_label": "Single-firm removal under fixed substitution",
        "disclaimer": "For research and policy analysis. Measures structural consequence, not probability.",
        "sample_data": False,
        "exported_at": date.today().isoformat(),
        "source": "cornell-semiconductor-project-main.zip",
        "available_firm_scenarios": firms_exported,
        "source_limitations": "The supplied archive contains report Top-25 tables and local case-study CSVs, not the full 78,257-row per-firm scenario export.",
    }


def build_source_coverage(scenarios: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "firms_exported": len(scenarios),
        "source_tables": sorted({scenario["source"]["table"] for scenario in scenarios}),
        "missing_from_archive": [
            "Full 78,257-firm firm-level scenario table",
            "Per-firm fragmentation values for Top-25 rows",
            "Per-firm backup-capacity-loss values for Top-25 rows",
            "Country/jurisdiction fields for Top-25 rows",
            "Per-firm affected-prime pathway counts except illustrative local case-study context",
        ],
    }


def build_export_stats(data_dir: Path, firms_dir: Path, firms_exported: int) -> Dict[str, Any]:
    scenario_files = sorted(firms_dir.glob("*.json"))
    largest = max(scenario_files, key=lambda path: path.stat().st_size) if scenario_files else None
    index_path = data_dir / "firms-index.json"
    return {
        "firms_exported": firms_exported,
        "scenario_files_written": len(scenario_files),
        "index_size_bytes": index_path.stat().st_size,
        "index_gzip_size_bytes": (data_dir / "firms-index.json.gz").stat().st_size,
        "total_output_size_bytes": sum(path.stat().st_size for path in data_dir.rglob("*") if path.is_file()),
        "largest_scenario_file": {
            "path": str(largest.relative_to(data_dir.parent)) if largest else None,
            "size_bytes": largest.stat().st_size if largest else 0,
        },
    }


def write_json(path: Path, payload: Any, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if pretty:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        else:
            json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
        handle.write("\n")


def write_gzip(source: Path, destination: Path) -> None:
    with source.open("rb") as input_handle, gzip.open(destination, "wb", compresslevel=9) as output_handle:
        shutil.copyfileobj(input_handle, output_handle)


if __name__ == "__main__":
    raise SystemExit(main())
