from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MATCH_TOLERANCE = 180
MEMORY_CLASS_MAP = {0.0: "Normal", 0.5: "Mild", 1.0: "Moderate/Severe", 2.0: "Moderate/Severe"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    train_dir = project_root / "scripts" / "train"

    tolerance = DEFAULT_MATCH_TOLERANCE
    output_stub = f"12-data_leakage_mri_clean_{tolerance}"

    parser = argparse.ArgumentParser(
        description=(
            "4-mr_path.csv ustunden MRI-level en yakin klinik eslesmeyi secip "
            "egitim icin temiz bir CSV kopyasi uretir. Orijinal dosyalara dokunmaz."
        )
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=script_dir / "4-mr_path.csv",
        help="MRI-path eslestirme CSV dosyasi.",
    )
    parser.add_argument(
        "--source-train-csv",
        type=Path,
        default=train_dir / "12-data_leakage.csv",
        help="Tum ozellikleri iceren mevcut egitim CSV dosyasi.",
    )
    parser.add_argument(
        "--output-match-csv",
        type=Path,
        default=script_dir / f"4-mr_path_clean_{tolerance}.csv",
        help="Temizlenmis MRI-path eslesme CSV cikti yolu.",
    )
    parser.add_argument(
        "--output-train-csv",
        type=Path,
        default=train_dir / f"{output_stub}.csv",
        help="Tek MRI-path iceren temiz egitim CSV cikti yolu.",
    )
    parser.add_argument(
        "--output-match-report-csv",
        type=Path,
        default=train_dir / f"{output_stub}_match_report.csv",
        help="Satir secim raporu CSV cikti yolu.",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=train_dir / f"{output_stub}_summary.json",
        help="Ozet rapor JSON cikti yolu.",
    )
    parser.add_argument(
        "--max-temporal-diff",
        type=int,
        default=tolerance,
        help="MRI session gunu ile klinik ziyaret arasindaki maksimum izin verilen fark.",
    )
    return parser.parse_args()


def parse_session_num_from_path(path_value: object) -> float:
    match = re.search(r"_d(\d+)$", str(path_value))
    if not match:
        return np.nan
    return float(match.group(1))


def coerce_memory_counts(memory_values: pd.Series) -> dict[str, int]:
    counts = {"Normal": 0, "Mild": 0, "Moderate/Severe": 0}
    for value in pd.to_numeric(memory_values, errors="coerce").fillna(0.0):
        counts[MEMORY_CLASS_MAP.get(float(value), "Normal")] += 1
    return counts


def build_best_mapping(mapping_df: pd.DataFrame, max_temporal_diff: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = mapping_df.copy()
    mapping = mapping.dropna(subset=["path"]).reset_index(drop=True)

    mapping["days_to_visit"] = pd.to_numeric(mapping["days_to_visit"], errors="coerce")
    mapping["memory"] = pd.to_numeric(mapping["memory"], errors="coerce")
    mapping["age at visit"] = pd.to_numeric(mapping["age at visit"], errors="coerce")
    mapping["session_num"] = pd.to_numeric(mapping.get("session_num"), errors="coerce")

    missing_session = mapping["session_num"].isna()
    mapping.loc[missing_session, "session_num"] = mapping.loc[missing_session, "path"].map(parse_session_num_from_path)

    mapping = mapping.dropna(subset=["days_to_visit", "memory", "age at visit", "session_num"]).reset_index(drop=True)
    mapping["temporal_diff"] = (mapping["days_to_visit"] - mapping["session_num"]).abs()

    mapping = mapping.sort_values(
        by=["path", "temporal_diff", "age at visit", "days_to_visit"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    best_per_path = mapping.groupby("path", sort=False, as_index=False).first()
    best_per_path = best_per_path[best_per_path["temporal_diff"] <= max_temporal_diff].reset_index(drop=True)
    return mapping, best_per_path


def choose_source_rows(
    source_df: pd.DataFrame,
    best_mapping_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    source = source_df.copy().reset_index(drop=True)
    source["_source_row_id"] = np.arange(len(source), dtype=np.int64)
    source["_memory_num"] = pd.to_numeric(source["memory"], errors="coerce")
    source["_age_num"] = pd.to_numeric(source["age at visit"], errors="coerce")

    path_to_indices: dict[str, list[int]] = {}
    for idx, path_value in enumerate(source["path"].astype(str)):
        path_to_indices.setdefault(path_value, []).append(idx)

    chosen_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    stats = {
        "missing_path_in_source": 0,
        "memory_mismatch_fallback": 0,
        "ambiguous_multirow_resolution": 0,
    }

    for mapping_row in best_mapping_df.to_dict("records"):
        path_value = str(mapping_row["path"])
        candidate_indices = path_to_indices.get(path_value, [])
        if not candidate_indices:
            stats["missing_path_in_source"] += 1
            continue

        candidates = source.iloc[candidate_indices].copy()
        candidates["memory_delta"] = (candidates["_memory_num"] - float(mapping_row["memory"])).abs()
        memory_matches = candidates[candidates["memory_delta"] < 1e-8].copy()
        resolution_type = "path+memory+closest_age"

        if memory_matches.empty:
            pool = candidates.copy()
            resolution_type = "path+closest_age_memory_fallback"
            stats["memory_mismatch_fallback"] += 1
        else:
            pool = memory_matches

        pool["age_delta"] = (pool["_age_num"] - float(mapping_row["age at visit"])).abs()
        pool = pool.sort_values(
            by=["age_delta", "_source_row_id"],
            ascending=[True, True],
            kind="mergesort",
        )
        selected = pool.iloc[0].copy()

        if len(candidates) > 1:
            stats["ambiguous_multirow_resolution"] += 1

        selected["selected_days_to_visit"] = float(mapping_row["days_to_visit"])
        selected["selected_session_num"] = float(mapping_row["session_num"])
        selected["selected_temporal_diff"] = float(mapping_row["temporal_diff"])
        selected["selected_mapping_age"] = float(mapping_row["age at visit"])
        selected["selected_mapping_memory"] = float(mapping_row["memory"])
        selected["resolution_type"] = resolution_type
        chosen_rows.append(selected.to_dict())

        match_rows.append(
            {
                "path": path_value,
                "oasis_id": mapping_row["OASISID"],
                "mapping_memory": float(mapping_row["memory"]),
                "mapping_age_at_visit": float(mapping_row["age at visit"]),
                "mapping_days_to_visit": float(mapping_row["days_to_visit"]),
                "mapping_session_num": float(mapping_row["session_num"]),
                "mapping_temporal_diff": float(mapping_row["temporal_diff"]),
                "candidate_count": int(len(candidates)),
                "memory_match_count": int(len(memory_matches)),
                "selected_source_row_id": int(selected["_source_row_id"]),
                "selected_source_age_at_visit": float(selected["_age_num"]),
                "selected_source_memory": float(selected["_memory_num"]),
                "selected_source_age_delta": float(selected["age_delta"]),
                "resolution_type": resolution_type,
            }
        )

    chosen_df = pd.DataFrame(chosen_rows)
    match_report_df = pd.DataFrame(match_rows)

    if chosen_df.empty:
        return chosen_df, match_report_df, stats

    helper_cols = ["_source_row_id", "_memory_num", "_age_num", "memory_delta", "age_delta"]
    chosen_df = chosen_df.drop(columns=[col for col in helper_cols if col in chosen_df.columns])
    chosen_df = chosen_df.sort_values(by="path", kind="mergesort").reset_index(drop=True)
    match_report_df = match_report_df.sort_values(by="path", kind="mergesort").reset_index(drop=True)
    return chosen_df, match_report_df, stats


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    mapping_df = pd.read_csv(args.mapping_csv)
    source_df = pd.read_csv(args.source_train_csv)

    full_mapping_df, best_mapping_df = build_best_mapping(mapping_df, max_temporal_diff=args.max_temporal_diff)
    clean_train_df, match_report_df, match_stats = choose_source_rows(source_df, best_mapping_df)

    if clean_train_df.empty:
        raise RuntimeError("Temiz egitim CSV olusturulamadi. Eslesen satir bulunamadi.")

    duplicate_paths = int(clean_train_df["path"].duplicated().sum())
    if duplicate_paths != 0:
        raise RuntimeError(f"Cikti CSV tekil degil: {duplicate_paths} duplicate path bulundu.")

    best_mapping_out = best_mapping_df.sort_values(by="path", kind="mergesort").reset_index(drop=True)

    for output_path in (
        args.output_match_csv,
        args.output_train_csv,
        args.output_match_report_csv,
        args.output_summary_json,
    ):
        ensure_parent(output_path)

    best_mapping_out.to_csv(args.output_match_csv, index=False)
    clean_train_df.to_csv(args.output_train_csv, index=False)
    match_report_df.to_csv(args.output_match_report_csv, index=False)

    summary = {
        "input": {
            "mapping_csv": str(args.mapping_csv),
            "source_train_csv": str(args.source_train_csv),
            "max_temporal_diff": int(args.max_temporal_diff),
        },
        "counts": {
            "mapping_rows_raw": int(len(mapping_df)),
            "mapping_rows_with_path": int(mapping_df["path"].notna().sum()),
            "mapping_unique_paths_raw": int(mapping_df["path"].dropna().nunique()),
            "mapping_rows_after_numeric_cleanup": int(len(full_mapping_df)),
            "best_mapping_rows_within_tolerance": int(len(best_mapping_df)),
            "source_rows": int(len(source_df)),
            "source_unique_paths": int(source_df["path"].nunique()),
            "output_rows": int(len(clean_train_df)),
            "output_unique_paths": int(clean_train_df["path"].nunique()),
        },
        "distributions": {
            "source_memory": source_df["memory"].value_counts(dropna=False).sort_index().to_dict(),
            "output_memory": clean_train_df["memory"].value_counts(dropna=False).sort_index().to_dict(),
            "output_merged_classes": coerce_memory_counts(clean_train_df["memory"]),
        },
        "temporal_diff": {
            "min": float(best_mapping_df["temporal_diff"].min()),
            "median": float(best_mapping_df["temporal_diff"].median()),
            "mean": float(best_mapping_df["temporal_diff"].mean()),
            "max": float(best_mapping_df["temporal_diff"].max()),
        },
        "resolution": {
            "missing_path_in_source": int(match_stats["missing_path_in_source"]),
            "memory_mismatch_fallback": int(match_stats["memory_mismatch_fallback"]),
            "ambiguous_multirow_resolution": int(match_stats["ambiguous_multirow_resolution"]),
            "resolution_type_counts": match_report_df["resolution_type"].value_counts().to_dict(),
        },
        "outputs": {
            "clean_mapping_csv": str(args.output_match_csv),
            "clean_train_csv": str(args.output_train_csv),
            "match_report_csv": str(args.output_match_report_csv),
            "summary_json": str(args.output_summary_json),
        },
    }

    with open(args.output_summary_json, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=True)

    print("=" * 72)
    print("MRI-level clean CSV build completed")
    print("=" * 72)
    print(f"Mapping raw rows: {len(mapping_df)}")
    print(f"Mapping unique paths: {mapping_df['path'].dropna().nunique()}")
    print(f"Selected MRI paths within {args.max_temporal_diff} days: {len(best_mapping_df)}")
    print(f"Source train rows: {len(source_df)}")
    print(f"Output clean rows: {len(clean_train_df)}")
    print(f"Output class counts: {summary['distributions']['output_merged_classes']}")
    print(
        "Temporal diff stats: "
        f"min={summary['temporal_diff']['min']:.0f}, "
        f"median={summary['temporal_diff']['median']:.0f}, "
        f"mean={summary['temporal_diff']['mean']:.1f}, "
        f"max={summary['temporal_diff']['max']:.0f}"
    )
    print(f"Clean mapping CSV : {args.output_match_csv}")
    print(f"Clean train CSV   : {args.output_train_csv}")
    print(f"Match report CSV  : {args.output_match_report_csv}")
    print(f"Summary JSON      : {args.output_summary_json}")


if __name__ == "__main__":
    main()
