from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate web/data list_by_type CSV from by_type regressions and k10-k1000 mapping."
    )
    parser.add_argument(
        "--by-type-dta",
        default="../output/wfd_regressions_by_type_17_fe.dta",
        help="Path to by_type .dta file containing sm_type.",
    )
    parser.add_argument(
        "--correspondence-dta",
        default="../data/k10_k1000_correspondence.dta",
        help="Path to k10_k1000 correspondence .dta file.",
    )
    parser.add_argument(
        "--k10-num",
        type=int,
        default=8,
        help="role_k10_v3_num value to map.",
    )
    parser.add_argument(
        "--output-csv",
        default="../web/data/list_by_type_17.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]

    by_type_path = project_dir / args.by_type_dta
    correspondence_path = project_dir / args.correspondence_dta
    output_path = project_dir / args.output_csv

    if not by_type_path.exists():
        raise FileNotFoundError(f"Missing by_type file: {by_type_path}")
    if not correspondence_path.exists():
        raise FileNotFoundError(f"Missing correspondence file: {correspondence_path}")

    by_type_df = pd.read_stata(by_type_path)
    if "sm_type" not in by_type_df.columns:
        raise ValueError(f"'sm_type' column not found in {by_type_path}")

    sm_types = by_type_df["sm_type"].dropna().astype(str).unique().tolist()
    if len(sm_types) == 0:
        raise ValueError(f"No non-missing sm_type values found in {by_type_path}")
    if len(sm_types) > 1:
        raise ValueError(f"Expected exactly one sm_type, found: {sm_types}")
    sm_type = sm_types[0]

    corr_df = pd.read_stata(correspondence_path)
    needed_cols = {"role_k10_v3_num", "role_k1000_v3_num"}
    missing_cols = needed_cols.difference(corr_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {correspondence_path}: {sorted(missing_cols)}")

    out_df = (
        corr_df.loc[corr_df["role_k10_v3_num"] == args.k10_num, ["role_k1000_v3_num"]]
        .dropna(subset=["role_k1000_v3_num"])
        .drop_duplicates()
        .sort_values("role_k1000_v3_num")
        .reset_index(drop=True)
    )
    out_df["role_k1000_v3_num"] = out_df["role_k1000_v3_num"].astype(int)
    out_df.insert(0, "sm_type", sm_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
