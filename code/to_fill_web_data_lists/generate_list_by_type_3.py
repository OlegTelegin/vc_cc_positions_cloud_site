from pathlib import Path

import pandas as pd


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]

    by_type_path = project_dir / "output" / "wfd_regressions_by_type_3_amir_category.dta"
    roles_path = project_dir / "web" / "data" / "amir_category_role_numbers.csv"
    output_path = project_dir / "web" / "data" / "list_by_type_3.csv"

    if not by_type_path.exists():
        raise FileNotFoundError(f"Missing file: {by_type_path}")
    if not roles_path.exists():
        raise FileNotFoundError(f"Missing file: {roles_path}")

    by_type_df = pd.read_stata(by_type_path)
    if "sm_type" not in by_type_df.columns:
        raise ValueError(f"'sm_type' column not found in {by_type_path}")

    sm_types = by_type_df["sm_type"].dropna().astype(str).unique().tolist()
    if len(sm_types) == 0:
        raise ValueError(f"No non-missing sm_type values found in {by_type_path}")
    if len(sm_types) > 1:
        raise ValueError(f"Expected exactly one sm_type, found: {sm_types}")
    sm_type = sm_types[0]

    roles_df = pd.read_csv(roles_path)
    if "role_k1000_v3_num" not in roles_df.columns:
        raise ValueError(f"'role_k1000_v3_num' column not found in {roles_path}")

    out_df = (
        roles_df[["role_k1000_v3_num"]]
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
