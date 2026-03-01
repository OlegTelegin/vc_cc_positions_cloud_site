from pathlib import Path

import pandas as pd


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]

    by_type_path = project_dir / "output" / "wfd_regressions_by_type_26_elastic_compensation_gbrl.dta"
    selected_weights_path = project_dir / "output" / "elasticnet_wfd_1702_6_compensation_selected_weights_nofe_supportaware.csv"
    output_path = project_dir / "web" / "data" / "list_by_type_26.csv"

    if not by_type_path.exists():
        raise FileNotFoundError(f"Missing file: {by_type_path}")
    if not selected_weights_path.exists():
        raise FileNotFoundError(f"Missing file: {selected_weights_path}")

    by_type_df = pd.read_stata(by_type_path)
    if "sm_type" not in by_type_df.columns:
        raise ValueError(f"'sm_type' column not found in {by_type_path}")
    sm_types = by_type_df["sm_type"].dropna().astype(str).unique().tolist()
    if len(sm_types) == 0:
        raise ValueError(f"No non-missing sm_type values found in {by_type_path}")
    if len(sm_types) > 1:
        raise ValueError(f"Expected exactly one sm_type, found: {sm_types}")
    sm_type = sm_types[0]

    selected_df = pd.read_csv(selected_weights_path)
    if "variable" not in selected_df.columns:
        raise ValueError(f"'variable' column not found in {selected_weights_path}")

    # Variables look like w_total_compensation_k1000_118; extract the k1000 numeric id.
    role_k1000 = (
        selected_df["variable"]
        .astype(str)
        .str.extract(r"^w_[a-z_]+_k1000_(\d+)$", expand=False)
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if role_k1000.empty:
        raise ValueError(f"No k1000 role numbers parsed from {selected_weights_path}")

    out_df = pd.DataFrame({"sm_type": sm_type, "role_k1000_v3_num": role_k1000})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
