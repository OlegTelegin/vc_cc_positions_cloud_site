from pathlib import Path
import re

import pandas as pd


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]

    do_file_path = project_dir / "code" / "run_wfd_regressions_by_type_23_miqp_gbrl_peremp_salary_1000_fe.do"
    selected_vars_path = project_dir / "data" / "miqp_gbrl_1000_peremp_salary_fe.csv"
    output_path = project_dir / "web" / "data" / "list_by_type_23.csv"

    if not do_file_path.exists():
        raise FileNotFoundError(f"Missing file: {do_file_path}")
    if not selected_vars_path.exists():
        raise FileNotFoundError(f"Missing file: {selected_vars_path}")

    do_text = do_file_path.read_text(encoding="utf-8")
    sm_type_match = re.search(r'^\s*local\s+sm_type\s+"([^"]+)"', do_text, flags=re.MULTILINE)
    if sm_type_match is None:
        raise ValueError(f'Could not parse `local sm_type "..."` from {do_file_path}')
    sm_type = sm_type_match.group(1)

    selected_df = pd.read_csv(selected_vars_path)
    if "selected_var" not in selected_df.columns:
        raise ValueError(f"'selected_var' column not found in {selected_vars_path}")

    # Values look like salary_k1000_21; extract the k1000 numeric id.
    role_k1000 = (
        selected_df["selected_var"]
        .astype(str)
        .str.extract(r"^[a-z_]+_k1000_(\d+)$", expand=False)
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if role_k1000.empty:
        raise ValueError(f"No k1000 role numbers parsed from {selected_vars_path}")

    out_df = pd.DataFrame({"sm_type": sm_type, "role_k1000_v3_num": role_k1000})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
