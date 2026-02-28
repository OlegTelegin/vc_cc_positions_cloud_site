from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_alpha_grid(y_train: np.ndarray, n_alphas: int = 200, ratio: float = 1e-6) -> np.ndarray:
    # For lasso: alpha_max gives all-zero coefficients; we trace down to alpha_max * ratio.
    y_centered = y_train - y_train.mean()
    alpha_max = np.max(np.abs(y_centered)) / max(len(y_train), 1)
    alpha_max = max(alpha_max, 1e-12)
    alpha_min = alpha_max * ratio
    return np.geomspace(alpha_max, alpha_min, n_alphas)


def load_allowed_suffixes(
    subset_path: Path,
    subset_col: str | None,
    min_suffix: int,
    max_suffix: int,
) -> list[int]:
    """
    Read a Stata .dta file containing allowed suffix numbers (e.g. 38, 39, 40),
    clean them, and return a sorted unique list restricted to [min_suffix, max_suffix].

    Rules:
    - If subset_col is given, use that column.
    - Else, if the file has exactly one column, use it.
    - Else, if there is exactly one numeric column, use it.
    - Otherwise, raise an error and ask the user to set subset_col explicitly.
    """

    df_sub = pd.read_stata(subset_path)

    if df_sub.shape[1] == 0:
        raise ValueError("Subset .dta file has no columns.")

    if subset_col is not None:
        if subset_col not in df_sub.columns:
            raise KeyError(
                f"subset_col='{subset_col}' not found in subset file.\n"
                f"Available columns: {list(df_sub.columns)}"
            )
        s = df_sub[subset_col]
    else:
        if df_sub.shape[1] == 1:
            s = df_sub.iloc[:, 0]
        else:
            numeric_cols = [c for c in df_sub.columns if pd.api.types.is_numeric_dtype(df_sub[c])]
            if len(numeric_cols) == 1:
                s = df_sub[numeric_cols[0]]
            else:
                raise ValueError(
                    "Could not auto-detect the subset-number column.\n"
                    "Please set subset_col explicitly.\n"
                    f"Columns in subset file: {list(df_sub.columns)}"
                )

    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        raise ValueError("Subset-number column is empty after numeric conversion.")

    vals = s.to_numpy(dtype=float)

    # Keep only integer-like values
    int_like = np.isfinite(vals) & (np.abs(vals - np.round(vals)) < 1e-9)
    vals = vals[int_like].astype(int)

    # Keep only values inside the valid suffix range
    vals = vals[(vals >= min_suffix) & (vals <= max_suffix)]

    allowed = sorted(set(vals.tolist()))
    if not allowed:
        raise ValueError(
            f"No valid suffixes remain after filtering to [{min_suffix}, {max_suffix}]."
        )

    return allowed


def fe_group_stats(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute group counts, group means of y, and group means of X.

    Parameters
    ----------
    X : (n, p) array
    y : (n,) array
    g : (n,) integer-coded groups in 0..n_groups-1
    n_groups : total number of possible groups

    Returns
    -------
    counts : (n_groups,) array
    y_means : (n_groups,) array
    X_means : (n_groups, p) array
    """
    counts = np.bincount(g, minlength=n_groups).astype(float)

    y_sums = np.bincount(g, weights=y, minlength=n_groups)
    y_means = np.zeros(n_groups, dtype=float)
    np.divide(y_sums, counts, out=y_means, where=counts > 0)

    X_sums = np.zeros((n_groups, X.shape[1]), dtype=float)
    np.add.at(X_sums, g, X)

    X_means = np.zeros_like(X_sums)
    nonzero = counts > 0
    X_means[nonzero] = X_sums[nonzero] / counts[nonzero, None]

    return counts, y_means, X_means


def fit_absorbed_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    alpha: float,
    n_groups: int,
) -> tuple[Pipeline, np.ndarray, np.ndarray, float]:
    """
    Fit nonnegative lasso with absorbed fixed effects (analogous to Stata i.group).

    The lasso is fit on within-transformed variables:
        y_tilde = y - E[y | group]
        X_tilde = X - E[X | group]

    Then we recover:
      - beta on the original X scale
      - per-group intercepts for prediction in the original scale
    """
    counts, y_means, X_means = fe_group_stats(X_train, y_train, g_train, n_groups)

    X_tilde = X_train - X_means[g_train]
    y_tilde = y_train - y_means[g_train]

    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("lasso", Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=1_000_000)),
        ]
    )
    model.fit(X_tilde, y_tilde)

    scaler = model.named_steps["scaler"]
    lasso = model.named_steps["lasso"]

    # Convert coefficients back to original X scale.
    beta_orig = lasso.coef_ / scaler.scale_

    # Recover group intercepts in the original scale:
    # E[y | g] = alpha_g + E[X | g] @ beta
    group_intercepts = np.full(n_groups, np.nan, dtype=float)
    seen = counts > 0
    if np.any(seen):
        group_intercepts[seen] = y_means[seen] - X_means[seen] @ beta_orig

    # Fallback if a prediction row belongs to a group unseen in training.
    fallback_intercept = float(np.mean(y_train - X_train @ beta_orig))

    return model, beta_orig, group_intercepts, fallback_intercept


def predict_with_absorbed_fe(
    X: np.ndarray,
    g: np.ndarray,
    beta_orig: np.ndarray,
    group_intercepts: np.ndarray,
    fallback_intercept: float,
) -> np.ndarray:
    """
    Predict in the original scale:
        y_hat = alpha_g + X @ beta
    using the learned per-group intercepts.
    """
    xb = X @ beta_orig
    alpha = group_intercepts[g]

    unseen = ~np.isfinite(alpha)
    if np.any(unseen):
        alpha = alpha.copy()
        alpha[unseen] = fallback_intercept

    return xb + alpha


def one_se_alpha_from_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    n_groups: int,
    alphas: np.ndarray,
    cv_splits: int = 10,
    seed: int = 1702,
) -> tuple[float, float]:
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    mse_path = np.zeros((len(alphas), cv_splits), dtype=float)

    for fold_idx, (fit_idx, val_idx) in enumerate(cv.split(X_train)):
        X_fit = X_train[fit_idx]
        y_fit = y_train[fit_idx]
        g_fit = g_train[fit_idx]

        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        g_val = g_train[val_idx]

        for alpha_idx, alpha in enumerate(alphas):
            _, beta_orig, group_intercepts, fallback_intercept = fit_absorbed_lasso(
                X_fit, y_fit, g_fit, alpha=alpha, n_groups=n_groups
            )
            y_hat = predict_with_absorbed_fe(
                X_val, g_val, beta_orig, group_intercepts, fallback_intercept
            )
            mse_path[alpha_idx, fold_idx] = np.mean((y_val - y_hat) ** 2)

    mse_mean = mse_path.mean(axis=1)
    mse_se = mse_path.std(axis=1, ddof=1) / np.sqrt(cv_splits)

    min_idx = int(np.argmin(mse_mean))
    threshold = mse_mean[min_idx] + mse_se[min_idx]

    # one-SE rule: largest penalty still within one SE of the best CV score
    candidate_idx = np.where(mse_mean <= threshold)[0]
    one_se_idx = int(candidate_idx[0]) if len(candidate_idx) > 0 else min_idx

    return float(alphas[min_idx]), float(alphas[one_se_idx])


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]

    # =========================
    # USER SETTINGS
    # =========================

    # Main data file
    data_path = project_dir / "data" / "temp_wfd_1702_6_w_certainty_quartiles_for_elastic_salary.dta"

    # Second .dta file containing the allowed variable numbers (e.g. 38, 39, 40, ...)
    subset_path = "D:/cursor_projects/vc_cc_positions_cloud_site/output/list_of_sm_related_k1000_gbrl.dta"

    # Column in subset_path that contains those numbers.
    # If None, the script auto-detects it.
    subset_col = None
    # Example:
    # subset_col = "num"

    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    yvar = "w_xsalesmktrawCIQ"
    fevar = "bucket16"

    # Predictor family
    x_prefix = "w_salary_k1000_"
    # x_prefix = "w_total_compensation_k1000_"
    x_min = 1
    x_max = 1004

    # =========================
    # CHECK FILES
    # =========================

    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    # Load allowed suffixes from the second .dta file
    allowed_suffixes = load_allowed_suffixes(
        subset_path=subset_path,
        subset_col=subset_col,
        min_suffix=x_min,
        max_suffix=x_max,
    )

    # Build restricted predictor list
    xvars = [f"{x_prefix}{i}" for i in allowed_suffixes]

    print(f"Loaded {len(allowed_suffixes)} allowed suffixes from: {subset_path}")
    print(f"Using restricted predictor set: {len(xvars)} variables")

    # =========================
    # LOAD DATA
    # =========================

    df = pd.read_stata(data_path).copy()

    # Match Stata split-sample behavior: split only complete cases for y + restricted xvars + FE.
    required_cols = [yvar, fevar] + xvars
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            "These required columns are missing from the main data file:\n"
            + "\n".join(missing_cols)
        )

    complete_mask = df[required_cols].notna().all(axis=1)
    complete_idx = df.index[complete_mask].to_numpy()

    if len(complete_idx) == 0:
        raise ValueError("No complete-case observations remain after applying y + FE + restricted xvars.")

    rng = np.random.default_rng(1702)
    n_complete = len(complete_idx)
    n_train = int(round(0.8 * n_complete))
    train_idx = rng.choice(complete_idx, size=n_train, replace=False)

    split = pd.Series(pd.NA, index=df.index, dtype="Int64")
    split.loc[complete_idx] = 2
    split.loc[train_idx] = 1
    df = df.assign(split=split)

    # Integer-code FE on the complete-case sample only.
    fe_code = pd.Series(pd.NA, index=df.index, dtype="Int64")
    fe_code.loc[complete_idx] = pd.Categorical(df.loc[complete_idx, fevar]).codes
    df = df.assign(fe_code=fe_code)

    train_mask = df["split"] == 1
    test_mask = df["split"] == 2

    X_train = df.loc[train_mask, xvars].to_numpy(dtype=float)
    y_train = df.loc[train_mask, yvar].to_numpy(dtype=float)
    g_train = df.loc[train_mask, "fe_code"].to_numpy(dtype=np.int64)

    X_test = df.loc[test_mask, xvars].to_numpy(dtype=float)
    y_test = df.loc[test_mask, yvar].to_numpy(dtype=float)
    g_test = df.loc[test_mask, "fe_code"].to_numpy(dtype=np.int64)

    n_groups = int(df.loc[complete_mask, "fe_code"].max()) + 1

    # =========================
    # CV + FIT
    # =========================

    # Build alpha grid off the within-transformed training y
    _, y_means_train, _ = fe_group_stats(X_train, y_train, g_train, n_groups)
    y_train_within = y_train - y_means_train[g_train]
    alphas = build_alpha_grid(y_train_within, n_alphas=200, ratio=1e-6)

    alpha_min, alpha_1se = one_se_alpha_from_cv(
        X_train, y_train, g_train, n_groups, alphas, cv_splits=10, seed=1702
    )

    _, coef_orig, group_intercepts, fallback_intercept = fit_absorbed_lasso(
        X_train, y_train, g_train, alpha=alpha_1se, n_groups=n_groups
    )

    # =========================
    # SELECTED VARIABLES
    # =========================

    selected_mask = coef_orig > 0.0
    selected_vars = [v for v, keep in zip(xvars, selected_mask) if keep]
    selected_weights = coef_orig[selected_mask]

    selected_df = pd.DataFrame({"variable": selected_vars, "weight": selected_weights})
    selected_df = selected_df.sort_values("variable").reset_index(drop=True)

    selected_path = output_dir / "elasticnet_wfd_1702_6_salary_selected_weights_gbrl_fe.csv"
    selected_df.to_csv(selected_path, index=False)

    # =========================
    # HOLDOUT PREDICTIONS
    # =========================

    yhat_test = predict_with_absorbed_fe(
        X_test, g_test, coef_orig, group_intercepts, fallback_intercept
    )

    pred_df = df.loc[test_mask, [yvar, fevar]].copy()
    pred_df["yhat"] = yhat_test
    pred_df["err"] = pred_df[yvar] - pred_df["yhat"]
    pred_df["abs_err"] = pred_df["err"].abs()
    pred_df["sq_err"] = pred_df["err"] ** 2

    pred_path = output_dir / "elasticnet_wfd_1702_6_salary_test_predictions_gbrl_fe.csv"
    pred_df.to_csv(pred_path, index=False)

    n_test = len(pred_df)
    rmse = float(np.sqrt(pred_df["sq_err"].mean())) if n_test > 0 else np.nan
    mae = float(pred_df["abs_err"].mean()) if n_test > 0 else np.nan

    if n_test > 1 and pred_df[yvar].std(ddof=1) > 0 and pred_df["yhat"].std(ddof=1) > 0:
        r2_holdout = float(np.corrcoef(pred_df[yvar], pred_df["yhat"])[0, 1] ** 2)
    else:
        r2_holdout = np.nan

    # =========================
    # TEXT SUMMARY
    # =========================

    results_path = output_dir / "elasticnet_wfd_1702_6_salary_results_gbrl_fe.txt"
    with results_path.open("w", encoding="utf-8") as fout:
        fout.write("Restricted nonnegative lasso run on temp_wfd_1702_6_w_certainty_quartiles.dta\n")
        fout.write(f"Dependent variable: {yvar}\n")
        fout.write(f"Fixed effects (absorbed, unpenalized): i.{fevar}\n")
        fout.write(f"Subset file: {subset_path}\n")
        fout.write(f"Allowed suffix count: {len(allowed_suffixes)}\n")
        fout.write(
            f"Predictors: restricted subset of {x_prefix}{x_min}-{x_prefix}{x_max}\n"
        )
        fout.write("Train/test split seed: 1702\n")
        fout.write(
            "Specification: nonnegative lasso (alpha=1) with absorbed bucket16 FE, "
            "CV + one-SE rule, grid(200, ratio(1e-6))\n"
        )
        fout.write(f"CV alpha(min-MSE): {alpha_min:.10g}\n")
        fout.write(f"CV alpha(one-SE): {alpha_1se:.10g}\n\n")

        fout.write("Allowed suffixes:\n")
        fout.write(" ".join(map(str, allowed_suffixes)) + "\n\n")

        fout.write("Selected variables:\n")
        if selected_vars:
            fout.write(" ".join(selected_vars) + "\n\n")
        else:
            fout.write("(none)\n\n")

        fout.write("Selected variables and weights (original X scale):\n")
        if not selected_df.empty:
            for _, row in selected_df.iterrows():
                fout.write(f"{row['variable']}\t{row['weight']:.8f}\n")
        else:
            fout.write("(none)\n")
        fout.write("\n")

        fout.write("Holdout metrics (split==2):\n")
        fout.write(f"N test = {n_test:10.0f}\n")
        fout.write(f"RMSE   = {rmse:12.4f}\n")
        fout.write(f"MAE    = {mae:12.4f}\n")
        fout.write(f"R2     = {r2_holdout:12.4f}\n")

    print(f"Saved: {selected_path}")
    print(f"Saved: {pred_path}")
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
