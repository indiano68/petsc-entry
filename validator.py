#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RANK_PREFIX_RE = re.compile(r"^\[(\d+)\]\s+")
MPI_PROCESS_RE = re.compile(r"\bMPI process(?:es)?\b")
IS_SIZE_RE = re.compile(r"^Number of indices in(?: \([^)]+\))? set (\d+)$")
L2G_RANGE_RE = re.compile(r"^(\d+):(\d+)\s+(\d+):(\d+)$")


class ValidationError(Exception):
    pass


@dataclass(frozen=True)
class ParsedFile:
    kind: str
    payload: object


def split_filename(path: Path) -> tuple[str, str, str]:
    parts = path.stem.rsplit("_", 2)
    if len(parts) != 3:
        raise ValidationError(
            f"{path.name}: expected filename format <name>_<classifier_1>_<classifier_2>"
        )
    return parts[0], parts[1], parts[2]


@dataclass(frozen=True)
class LocalToGlobalMapping:
    process_count: int
    normalized: list[int]


def parse_petsc_ascii(path: Path) -> ParsedFile:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    if not lines:
        raise ValidationError(f"{path.name}: empty file")

    header = lines[0].strip()
    if header.startswith("Mat Object:"):
        return ParsedFile("mat", parse_mat(lines, path))
    if header.startswith("IS Object:"):
        return ParsedFile("is", parse_is(lines, path))
    if header.startswith("Vec Object:"):
        return ParsedFile("vec", parse_vec(lines, path))
    if header.startswith("ISLocalToGlobalMapping Object:"):
        return ParsedFile("l2g", parse_l2g(lines, path))

    raise ValidationError(f"{path.name}: unsupported PETSc ASCII object header {header!r}")


def parse_mat(lines: list[str], path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    expected_width: int | None = None

    for raw_line in lines[1:]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("type:"):
            continue
        if MPI_PROCESS_RE.search(stripped):
            continue
        if stripped.startswith("["):
            stripped = RANK_PREFIX_RE.sub("", stripped)
        try:
            row = [float(token) for token in stripped.split()]
        except ValueError as exc:
            raise ValidationError(f"{path.name}: invalid numeric matrix row {raw_line!r}") from exc
        if not row:
            continue
        if expected_width is None:
            expected_width = len(row)
        elif len(row) != expected_width:
            raise ValidationError(
                f"{path.name}: inconsistent matrix row width {len(row)} != {expected_width}"
            )
        rows.append(row)

    if not rows:
        raise ValidationError(f"{path.name}: no matrix rows found")

    return rows


def parse_is(lines: list[str], path: Path) -> list[int]:
    values: list[int] = []
    declared_total = 0

    for raw_line in lines[1:]:
        stripped = raw_line.strip()
        if not stripped:
            continue

        stripped = RANK_PREFIX_RE.sub("", stripped)
        if stripped.startswith("type:"):
            continue
        size_match = IS_SIZE_RE.match(stripped)
        if size_match:
            try:
                declared_total += int(size_match.group(1))
            except ValueError as exc:
                raise ValidationError(
                    f"{path.name}: invalid IS size declaration {raw_line!r}"
                ) from exc
            continue

        pieces = stripped.split()
        if len(pieces) != 2:
            raise ValidationError(f"{path.name}: invalid IS entry {raw_line!r}")
        try:
            int(pieces[0])
            value = int(pieces[1])
        except ValueError as exc:
            raise ValidationError(f"{path.name}: invalid IS entry {raw_line!r}") from exc
        values.append(value)

    if not values:
        raise ValidationError(f"{path.name}: no IS entries found")

    if declared_total and declared_total != len(values):
        raise ValidationError(
            f"{path.name}: declared {declared_total} indices but parsed {len(values)}"
        )

    return values


def parse_vec(lines: list[str], path: Path) -> list[float]:
    values: list[float] = []

    for raw_line in lines[1:]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("type:"):
            continue
        if stripped.startswith("Process ["):
            continue
        if MPI_PROCESS_RE.search(stripped):
            continue
        if stripped.startswith("["):
            stripped = RANK_PREFIX_RE.sub("", stripped)
        try:
            values.append(float(stripped))
        except ValueError as exc:
            raise ValidationError(f"{path.name}: invalid vector entry {raw_line!r}") from exc

    return values


def parse_l2g(lines: list[str], path: Path) -> LocalToGlobalMapping:
    header = lines[0].strip()
    match = re.search(r":\s+(\d+)\s+MPI process(?:es)?$", header)
    if match is None:
        raise ValidationError(f"{path.name}: invalid ISLocalToGlobal header {header!r}")

    process_count = int(match.group(1))
    by_rank: dict[int, list[tuple[int, int, int, int]]] = {}

    for raw_line in lines[1:]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("type"):
            continue
        if MPI_PROCESS_RE.search(stripped):
            continue
        rank_match = RANK_PREFIX_RE.match(stripped)
        if rank_match is None:
            raise ValidationError(f"{path.name}: missing rank prefix in {raw_line!r}")
        rank = int(rank_match.group(1))
        stripped = RANK_PREFIX_RE.sub("", stripped)

        range_match = L2G_RANGE_RE.match(stripped)
        if range_match is None:
            raise ValidationError(f"{path.name}: invalid ISLocalToGlobal entry {raw_line!r}")

        local_start, local_end, global_start, global_end = (
            int(range_match.group(1)),
            int(range_match.group(2)),
            int(range_match.group(3)),
            int(range_match.group(4)),
        )
        local_count = local_end - local_start
        global_count = global_end - global_start
        if local_count < 0 or global_count < 0:
            raise ValidationError(f"{path.name}: descending ISLocalToGlobal range {raw_line!r}")
        if local_count != global_count:
            raise ValidationError(
                f"{path.name}: inconsistent ISLocalToGlobal range lengths in {raw_line!r}"
            )

        by_rank.setdefault(rank, []).append((local_start, local_end, global_start, global_end))

    if not by_rank:
        raise ValidationError(f"{path.name}: no ISLocalToGlobal entries found")

    normalized: list[int] = []
    for rank in range(process_count):
        entries = by_rank.get(rank, [])
        if not entries:
            continue

        rank_values: list[int | None] = []
        expected_local_start = 0
        for local_start, local_end, global_start, global_end in entries:
            if local_start != expected_local_start:
                raise ValidationError(
                    f"{path.name}: non-contiguous local range on rank {rank}: "
                    f"expected {expected_local_start}, got {local_start}"
                )

            local_count = local_end - local_start
            global_count = global_end - global_start
            if local_count != global_count:
                raise ValidationError(
                    f"{path.name}: inconsistent ISLocalToGlobal range lengths on rank {rank}"
                )

            rank_values.extend(range(global_start, global_end))
            expected_local_start = local_end

        normalized.extend(rank_values)

    return LocalToGlobalMapping(process_count=process_count, normalized=normalized)


def compare_parsed(
    left: ParsedFile,
    right: ParsedFile,
    rel_tol: float,
    abs_tol: float,
    left_classifier_1: str | None = None,
    right_classifier_1: str | None = None,
) -> str | None:
    if left.kind != right.kind:
        return f"type mismatch: {left.kind} != {right.kind}"
    if left.kind == "mat":
        return compare_mats(left.payload, right.payload, rel_tol, abs_tol)
    if left.kind == "is":
        return compare_index_sets(left.payload, right.payload)
    if left.kind == "vec":
        return compare_vecs(left.payload, right.payload, rel_tol, abs_tol)
    if left.kind == "l2g":
        return compare_l2g(left.payload, right.payload, left_classifier_1, right_classifier_1)
    return f"unsupported parsed kind {left.kind!r}"


def compare_mats(
    left: list[list[float]],
    right: list[list[float]],
    rel_tol: float,
    abs_tol: float,
) -> str | None:
    if len(left) != len(right):
        return f"row count mismatch: {len(left)} != {len(right)}"
    if left and right and len(left[0]) != len(right[0]):
        return f"column count mismatch: {len(left[0])} != {len(right[0])}"

    for i, (left_row, right_row) in enumerate(zip(left, right)):
        if len(left_row) != len(right_row):
            return f"row {i} width mismatch: {len(left_row)} != {len(right_row)}"
        for j, (left_value, right_value) in enumerate(zip(left_row, right_row)):
            if not math.isclose(left_value, right_value, rel_tol=rel_tol, abs_tol=abs_tol):
                return (
                    f"matrix value mismatch at ({i}, {j}): "
                    f"{left_value} != {right_value}"
                )
    return None


def compare_index_sets(left: list[int], right: list[int]) -> str | None:
    if len(left) != len(right):
        return f"index count mismatch: {len(left)} != {len(right)}"
    for i, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return f"index mismatch at position {i}: {left_value} != {right_value}"
    return None


def compare_vecs(
    left: list[float],
    right: list[float],
    rel_tol: float,
    abs_tol: float,
) -> str | None:
    if len(left) != len(right):
        return f"vector length mismatch: {len(left)} != {len(right)}"
    for i, (left_value, right_value) in enumerate(zip(left, right)):
        if not math.isclose(left_value, right_value, rel_tol=rel_tol, abs_tol=abs_tol):
            return f"vector value mismatch at {i}: {left_value} != {right_value}"
    return None


def compare_l2g(
    left: LocalToGlobalMapping,
    right: LocalToGlobalMapping,
    left_classifier_1: str | None,
    right_classifier_1: str | None,
) -> str | None:
    _ = (left_classifier_1, right_classifier_1)
    if len(left.normalized) != len(right.normalized):
        return f"mapping length mismatch: {len(left.normalized)} != {len(right.normalized)}"
    for i, (left_value, right_value) in enumerate(zip(left.normalized, right.normalized)):
        if left_value != right_value:
            return (
                f"normalized mapping mismatch at global-local index {i}: "
                f"{left_value} != {right_value}"
            )
    return None


def classifier_sort_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def iter_groups(paths: Iterable[Path]) -> dict[tuple[str, str], list[Path]]:
    groups: dict[tuple[str, str], list[Path]] = {}
    for path in paths:
        try:
            name, classifier_1, classifier_2 = split_filename(path)
        except ValidationError:
            continue
        groups.setdefault((name, classifier_2), []).append(path)
    for group_paths in groups.values():
        group_paths.sort(key=lambda path: classifier_sort_key(split_filename(path)[1]))
    return groups


def validate_directory(directory: Path, rel_tol: float, abs_tol: float) -> int:
    if not directory.is_dir():
        raise ValidationError(f"{directory}: directory does not exist")

    files = sorted(path for path in directory.iterdir() if path.is_file())
    groups = iter_groups(files)
    if not groups:
        raise ValidationError(f"{directory}: no files to validate")

    failures = 0
    parse_errors = 0
    compared_groups = 0

    for (name, classifier_2), group_paths in sorted(groups.items()):
        if len(group_paths) < 2:
            continue

        parsed_group: list[tuple[Path, ParsedFile]] = []
        for path in group_paths:
            try:
                parsed_group.append((path, parse_petsc_ascii(path)))
            except ValidationError as exc:
                parse_errors += 1
                print(f"ERROR {exc}")

        if len(parsed_group) < 2:
            continue

        compared_groups += 1
        reference_path, parsed_reference = parsed_group[0]
        _, reference_classifier_1, _ = split_filename(reference_path)
        group_failed = False

        for candidate_path, parsed_candidate in parsed_group[1:]:
            _, candidate_classifier_1, _ = split_filename(candidate_path)
            mismatch = compare_parsed(
                parsed_reference,
                parsed_candidate,
                rel_tol,
                abs_tol,
                reference_classifier_1,
                candidate_classifier_1,
            )
            if mismatch is not None:
                group_failed = True
                failures += 1
                print(
                    f"FAIL {name} classifier_2={classifier_2}: "
                    f"{reference_path.name} vs {candidate_path.name}: {mismatch}"
                )

        if not group_failed:
            members = ", ".join(path.name for path, _ in parsed_group)
            print(f"OK   {name} classifier_2={classifier_2}: {members}")

    if compared_groups == 0:
        print("No comparable groups found.")
        return 0

    if failures or parse_errors:
        print(
            "\nValidation failed: "
            f"{failures} mismatched comparison(s), {parse_errors} parse error(s)."
        )
        return 1

    print(f"\nValidation passed: {compared_groups} group(s) matched.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PETSc ASCII outputs grouped by <name> and <classifier_2> "
            "for files named <name>_<classifier_1>_<classifier_2>.*"
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="outputs",
        help="directory containing PETSc ASCII files (default: outputs)",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-9,
        help="relative tolerance for matrix comparisons (default: 1e-9)",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-12,
        help="absolute tolerance for matrix comparisons (default: 1e-12)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        return validate_directory(Path(args.directory), args.rel_tol, args.abs_tol)
    except ValidationError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
