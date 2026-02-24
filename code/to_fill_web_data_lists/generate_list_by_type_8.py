from pathlib import Path

import pandas as pd


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]

    by_type_path = project_dir / "output" / "wfd_regressions_by_type_8_selected_vars_salary_1000.dta"
    selected_vars_path = project_dir / "data" / "selected_vars_predict_salary_w_salary_1000.csv"
    output_path = project_dir / "web" / "data" / "list_by_type_8.csv"

    if not by_type_path.exists():
        raise FileNotFoundError(f"Missing file: {by_type_path}")
    if not selected_vars_path.exists():
        raise FileNotFoundError(f"Missing file: {selected_vars_path}")

    by_type_df = pd.read_stata(by_type_path)
    if "sm_type" not in by_type_df.columns:
        raise ValueError(f"'sm_type' column not found in {by_type_path}")
    sm_types = by_type_df["sm_type"].dropna().astype(str).unique().tolist()
    if len(sm_types) == 0:
        raise ValueError(f"No non-missing sm_type values found in {by_type_path}")
    if len(sm_types) > 1:
        raise ValueError(f"Expected exactly one sm_type, found: {sm_types}")
    sm_type = sm_types[0]

    selected_df = pd.read_csv(selected_vars_path)
    if "selected_var" not in selected_df.columns:
        raise ValueError(f"'selected_var' column not found in {selected_vars_path}")

    # Values look like salary_k1000_633; extract the k1000 numeric id.
    k1000_ids = (
        selected_df["selected_var"]
        .astype(str)
        .str.extract(r"^[a-z_]+_k1000_(\d+)$", expand=False)
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if k1000_ids.empty:
        raise ValueError(f"No k1000 role numbers parsed from {selected_vars_path}")

    out_df = pd.DataFrame({"sm_type": sm_type, "role_k1000_v3_num": k1000_ids})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
