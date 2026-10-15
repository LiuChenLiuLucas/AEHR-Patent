#!/usr/bin/env python3
"""Audit patent-classification TSV splits using only the Python standard library."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


CPC_RE = re.compile(r"^[A-HY][0-9]{2}[A-Z]$")
ID_COLUMNS = ("patent_id", "id", "publication_number", "doc_id")
LABEL_COLUMNS = ("labels", "cpc", "cpc_labels", "subclass")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value: Any) -> str:
    token = str(value or "").strip().upper().split("/")[0]
    token = re.sub(r"[^A-Z0-9]", "", token)
    return token[:4] if len(token) >= 4 else token


def parse_labels(raw: str) -> list[str]:
    value = (raw or "").strip()
    if not value:
        return []
    parsed: Any = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            break
        except (ValueError, SyntaxError, json.JSONDecodeError):
            pass
    if isinstance(parsed, dict):
        parsed = parsed.get("labels", [])
    if isinstance(parsed, (list, tuple, set)):
        values = parsed
    elif parsed is not None:
        values = [parsed]
    else:
        values = re.split(r"[,;|]", value)
    return list(dict.fromkeys(label for item in values if (label := normalize_label(item))))


def pick_column(headers: list[str], candidates: tuple[str, ...], required: bool) -> str | None:
    by_lower = {header.strip().lower(): header for header in headers}
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    if required:
        raise ValueError(f"Missing expected column; looked for {', '.join(candidates)}; found {headers}")
    return None


def audit_split(path: Path) -> tuple[dict[str, Any], set[str], set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        headers = reader.fieldnames or []
        id_column = pick_column(headers, ID_COLUMNS, required=False)
        title_column = pick_column(headers, ("title",), required=False)
        abstract_column = pick_column(headers, ("abstract",), required=False)
        labels_column = pick_column(headers, LABEL_COLUMNS, required=True)
        ids: set[str] = set()
        text_hashes: set[str] = set()
        label_counts: Counter[str] = Counter()
        duplicate_ids = duplicate_texts = 0
        empty_titles = empty_abstracts = empty_text_rows = empty_label_rows = 0
        rows_with_invalid_labels = malformed_label_rows = 0
        invalid_labels: Counter[str] = Counter()
        row_count = 0
        for row in reader:
            row_count += 1
            raw_id = (row.get(id_column) or "").strip() if id_column else ""
            if raw_id:
                normalized_id = raw_id.upper()
                duplicate_ids += normalized_id in ids
                ids.add(normalized_id)
            title = (row.get(title_column) or "").strip() if title_column else ""
            abstract = (row.get(abstract_column) or "").strip() if abstract_column else ""
            empty_titles += bool(title_column and not title)
            empty_abstracts += bool(abstract_column and not abstract)
            text = f"{title} {abstract}".strip().lower()
            if text:
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                duplicate_texts += text_hash in text_hashes
                text_hashes.add(text_hash)
            else:
                empty_text_rows += 1
            raw_labels = row.get(labels_column) or ""
            labels = parse_labels(raw_labels)
            if not labels:
                empty_label_rows += 1
                malformed_label_rows += bool(raw_labels.strip())
            invalid = [label for label in labels if not CPC_RE.fullmatch(label)]
            if invalid:
                rows_with_invalid_labels += 1
                invalid_labels.update(invalid)
            label_counts.update(label for label in labels if CPC_RE.fullmatch(label))

    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": row_count,
        "headers": headers,
        "columns": {"id": id_column, "title": title_column, "abstract": abstract_column, "labels": labels_column},
        "id_rows_present": len(ids),
        "duplicate_id_count": duplicate_ids,
        "duplicate_normalized_text_count": duplicate_texts,
        "empty_title_rows": empty_titles,
        "empty_abstract_rows": empty_abstracts,
        "empty_text_rows": empty_text_rows,
        "empty_label_rows": empty_label_rows,
        "malformed_label_rows": malformed_label_rows,
        "rows_with_invalid_cpc_labels": rows_with_invalid_labels,
        "invalid_labels_top20": invalid_labels.most_common(20),
        "unique_labels": len(label_counts),
        "label_assignments": sum(label_counts.values()),
        "label_frequency": dict(sorted(label_counts.items())),
    }
    return report, ids, text_hashes


def audit_directory(data_dir: Path) -> dict[str, Any]:
    names = ("train.tsv", "dev.tsv", "test.tsv")
    split_details = {name.removesuffix(".tsv"): audit_split(data_dir / name) for name in names}
    split_names = list(split_details)
    overlaps: dict[str, dict[str, int]] = {}
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            left_ids, left_texts = split_details[left][1:]
            right_ids, right_texts = split_details[right][1:]
            overlaps[f"{left}__{right}"] = {
                "id_overlap": len(left_ids & right_ids),
                "exact_normalized_text_overlap": len(left_texts & right_texts),
            }
    return {
        "data_dir": str(data_dir),
        "splits": {name: details[0] for name, details in split_details.items()},
        "cross_split_overlaps": overlaps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing train.tsv, dev.tsv, test.tsv")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = audit_directory(args.data_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    overlaps = report["cross_split_overlaps"]
    has_overlap = any(item["id_overlap"] or item["exact_normalized_text_overlap"] for item in overlaps.values())
    return 2 if has_overlap else 0


if __name__ == "__main__":
    sys.exit(main())
