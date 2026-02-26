from pathlib import Path

import pandas as pd


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]

    by_type_path = project_dir / "output" / "wfd_regressions_by_type_19_elastic_compensation_fe.dta"
    selected_weights_path = project_dir / "output" / "elasticnet_wfd_1702_6_total_compensation_selected_weights_fe.csv"
    correspondence_path = project_dir / "data" / "k50_k1000_correspondence.dta"
    output_path = project_dir / "web" / "data" / "list_by_type_19.csv"

    if not by_type_path.exists():
        raise FileNotFoundError(f"Missing file: {by_type_path}")
    if not selected_weights_path.exists():
        raise FileNotFoundError(f"Missing file: {selected_weights_path}")
    if not correspondence_path.exists():
        raise FileNotFoundError(f"Missing file: {correspondence_path}")

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

    # Variables look like w_total_compensation_k50_20; extract the k50 numeric id.
    k50_ids = (
        selected_df["variable"]
        .astype(str)
        .str.extract(r"^w_[a-z_]+_k50_(\d+)$", expand=False)
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if len(k50_ids) == 0:
        raise ValueError(f"No k50 role numbers parsed from {selected_weights_path}")

    corr_df = pd.read_stata(correspondence_path)
    required_cols = {"role_k50_v3_num", "role_k1000_v3_num"}
    missing_cols = required_cols.difference(corr_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {correspondence_path}: {sorted(missing_cols)}")

    mapped_df = corr_df.loc[
        corr_df["role_k50_v3_num"].isin(k50_ids), ["role_k1000_v3_num"]
    ].dropna(subset=["role_k1000_v3_num"])

    role_k1000 = (
        mapped_df["role_k1000_v3_num"]
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if role_k1000.empty:
        raise ValueError("No role_k1000_v3_num values found after mapping from selected k50 ids.")

    out_df = pd.DataFrame({"sm_type": sm_type, "role_k1000_v3_num": role_k1000})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
