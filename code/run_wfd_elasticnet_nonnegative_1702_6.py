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


def one_se_alpha_from_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: np.ndarray,
    cv_splits: int = 10,
    seed: int = 1702,
) -> tuple[float, float]:
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    mse_path = np.zeros((len(alphas), cv_splits), dtype=float)

    for fold_idx, (fit_idx, val_idx) in enumerate(cv.split(X_train)):
        X_fit = X_train[fit_idx]
        y_fit = y_train[fit_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        for alpha_idx, alpha in enumerate(alphas):
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lasso", Lasso(alpha=alpha, positive=True, max_iter=1_000_000)),
                ]
            )
            model.fit(X_fit, y_fit)
            y_hat = model.predict(X_val)
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
    data_path = project_dir / "data" / "temp_wfd_1702_6.dta"
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    yvar = "w_xsalesmktrawCIQ"
    xvars = [f"w_salary_k50_{i}" for i in range(1, 54)]

    df = pd.read_stata(data_path).copy()

    # Match Stata splitsample behavior: split only complete cases for y + xvars.
    required_cols = [yvar] + xvars
    complete_mask = df[required_cols].notna().all(axis=1)
    complete_idx = df.index[complete_mask].to_numpy()

    rng = np.random.default_rng(1702)
    n_complete = len(complete_idx)
    n_train = int(round(0.8 * n_complete))
    train_idx = rng.choice(complete_idx, size=n_train, replace=False)

    split = pd.Series(pd.NA, index=df.index, dtype="Int64")
    split.loc[complete_idx] = 2
    split.loc[train_idx] = 1
    df = df.assign(split=split)

    train_mask = df["split"] == 1
    test_mask = df["split"] == 2

    X_train = df.loc[train_mask, xvars].to_numpy(dtype=float)
    y_train = df.loc[train_mask, yvar].to_numpy(dtype=float)
    X_test = df.loc[test_mask, xvars].to_numpy(dtype=float)
    y_test = df.loc[test_mask, yvar].to_numpy(dtype=float)

    alphas = build_alpha_grid(y_train, n_alphas=200, ratio=1e-6)
    alpha_min, alpha_1se = one_se_alpha_from_cv(X_train, y_train, alphas, cv_splits=10, seed=1702)

    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=alpha_1se, positive=True, max_iter=1_000_000)),
        ]
    )
    final_model.fit(X_train, y_train)

    lasso = final_model.named_steps["lasso"]
    coef = lasso.coef_
    selected_mask = coef > 0.0
    selected_vars = [v for v, keep in zip(xvars, selected_mask) if keep]
    selected_weights = coef[selected_mask]

    selected_df = pd.DataFrame({"variable": selected_vars, "weight": selected_weights})
    selected_df = selected_df.sort_values("variable").reset_index(drop=True)
    selected_df.to_csv(output_dir / "elasticnet_wfd_1702_6_selected_weights.csv", index=False)

    yhat_test = final_model.predict(X_test)
    pred_df = df.loc[test_mask, [yvar]].copy()
    pred_df["yhat"] = yhat_test
    pred_df["err"] = pred_df[yvar] - pred_df["yhat"]
    pred_df["abs_err"] = pred_df["err"].abs()
    pred_df["sq_err"] = pred_df["err"] ** 2
    pred_df.to_csv(output_dir / "elasticnet_wfd_1702_6_test_predictions.csv", index=False)

    n_test = len(pred_df)
    rmse = float(np.sqrt(pred_df["sq_err"].mean())) if n_test > 0 else np.nan
    mae = float(pred_df["abs_err"].mean()) if n_test > 0 else np.nan
    if n_test > 1 and pred_df[yvar].std(ddof=1) > 0 and pred_df["yhat"].std(ddof=1) > 0:
        r2_holdout = float(np.corrcoef(pred_df[yvar], pred_df["yhat"])[0, 1] ** 2)
    else:
        r2_holdout = np.nan

    with (output_dir / "elasticnet_wfd_1702_6_results.txt").open("w", encoding="utf-8") as fout:
        fout.write("Elastic net run on temp_wfd_1702_6.dta\n")
        fout.write(f"Dependent variable: {yvar}\n")
        fout.write("Predictors: w_salary_k50_1-w_salary_k50_53\n")
        fout.write("Train/test split seed: 1702\n")
        fout.write(
            "Specification: nonnegative lasso (alpha=1) with CV + one-SE rule, "
            "grid(200, ratio(1e-6))\n"
        )
        fout.write(f"CV alpha(min-MSE): {alpha_min:.10g}\n")
        fout.write(f"CV alpha(one-SE): {alpha_1se:.10g}\n\n")
        fout.write("Selected variables:\n")
        fout.write(" ".join(selected_vars) + "\n\n")
        fout.write("Selected variables and weights:\n")
        for _, row in selected_df.iterrows():
            fout.write(f"{row['variable']}\t{row['weight']:.8f}\n")
        fout.write("\n")
        fout.write("Holdout metrics (split==2):\n")
        fout.write(f"N test = {n_test:10.0f}\n")
        fout.write(f"RMSE   = {rmse:12.4f}\n")
        fout.write(f"MAE    = {mae:12.4f}\n")
        fout.write(f"R2     = {r2_holdout:12.4f}\n")

    print(f"Saved: {output_dir / 'elasticnet_wfd_1702_6_selected_weights.csv'}")
    print(f"Saved: {output_dir / 'elasticnet_wfd_1702_6_test_predictions.csv'}")
    print(f"Saved: {output_dir / 'elasticnet_wfd_1702_6_results.txt'}")


if __name__ == "__main__":
    main()
