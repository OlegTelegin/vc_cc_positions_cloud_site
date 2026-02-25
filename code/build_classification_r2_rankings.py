from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


TYPE_RE = re.compile(r"^list_by_type_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build classification_r2_rankings.csv from classification_sources.csv and output DTA files."
    )
    parser.add_argument(
        "--sources",
        default="web/data/classification_sources.csv",
        help="Path to classification_sources.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory containing wfd_regressions_by_type_*.dta files",
    )
    parser.add_argument(
        "--dest",
        default="web/data/classification_r2_rankings.csv",
        help="Destination CSV path",
    )
    return parser.parse_args()


def resolve_regression_file(file_name: str, output_dir: Path) -> Path:
    match = TYPE_RE.match(file_name.strip())
    if not match:
        raise ValueError(f"Unsupported file_name format: {file_name!r}")

    type_num = match.group(1)
    # Match only the exact type number. This avoids type 1 matching 10/11/12/13.
    exact = output_dir.glob(f"wfd_regressions_by_type_{type_num}.dta")
    with_suffix = output_dir.glob(f"wfd_regressions_by_type_{type_num}_*.dta")
    candidates = sorted({*exact, *with_suffix})
    if not candidates:
        raise FileNotFoundError(
            f"No regression result file found for {file_name!r} in {output_dir}"
        )

    # Prefer latest modified file if multiple candidates exist.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_one_result(file_name: str, result_path: Path) -> pd.DataFrame:
    df = pd.read_stata(result_path)
    required = {"regression_number", "r2", "coefficient"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{result_path} is missing expected columns: {sorted(missing)}"
        )

    out = df.loc[:, ["regression_number", "r2", "coefficient"]].copy()
    out["file_name"] = file_name
    return out


def main() -> None:
    args = parse_args()
    sources_path = Path(args.sources)
    output_dir = Path(args.output_dir)
    dest_path = Path(args.dest)

    sources = pd.read_csv(sources_path)
    if "file_name" not in sources.columns:
        raise ValueError(f"{sources_path} must contain a 'file_name' column")

    frames: list[pd.DataFrame] = []
    for file_name in sources["file_name"].dropna().astype(str):
        result_path = resolve_regression_file(file_name, output_dir)
        frames.append(load_one_result(file_name, result_path))

    if not frames:
        raise ValueError("No classification entries found to process.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[:, ["regression_number", "file_name", "r2", "coefficient"]]
    combined["regression_number"] = pd.to_numeric(
        combined["regression_number"], errors="coerce"
    )
    combined = combined.dropna(subset=["regression_number", "r2"])
    combined["regression_number"] = combined["regression_number"].astype(int)

    combined = combined.sort_values(
        by=["regression_number", "r2"], ascending=[True, False]
    ).reset_index(drop=True)
    combined["rank"] = combined.groupby("regression_number").cumcount() + 1
    combined = combined.loc[:, ["regression_number", "file_name", "rank", "r2", "coefficient"]]

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(dest_path, index=False)
    print(f"Wrote {len(combined)} rows to {dest_path}")


if __name__ == "__main__":
    main()
