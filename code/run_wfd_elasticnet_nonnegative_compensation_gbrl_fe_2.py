from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# LOGGING
# =========================================================

def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


# =========================================================
# HELPERS
# =========================================================

def load_allowed_suffixes(
    subset_path: Path | str,
    subset_col: str | None,
    min_suffix: int,
    max_suffix: int,
) -> list[int]:
    """
    Read a Stata .dta file containing allowed suffix numbers (e.g. 38, 39, 40),
    clean them, and return a sorted unique list restricted to [min_suffix, max_suffix].
    """
    subset_path = Path(subset_path)

    if not subset_path.exists():
        raise FileNotFoundError(f"Cannot find subset file: {subset_path}")

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


def within_transform(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Within-transform for absorbed FE:
        X_tilde = X - E[X | group]
        y_tilde = y - E[y | group]
    """
    _, y_means, X_means = fe_group_stats(X, y, g, n_groups)
    X_tilde = X - X_means[g]
    y_tilde = y - y_means[g]
    return X_tilde, y_tilde


def build_alpha_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    n_groups: int,
    model_kind: str,
    l1_ratio: float,
    n_alphas: int = 400,
    ratio: float = 1e-10,
) -> np.ndarray:
    """
    Build an alpha grid from the within-transformed training data.
    """
    X_tilde, y_tilde = within_transform(X_train, y_train, g_train, n_groups)

    # Approximate the pipeline's scaling when deriving alpha_max
    scale = np.sqrt(np.mean(X_tilde**2, axis=0))
    bad_scale = ~np.isfinite(scale) | (scale <= 1e-12)
    scale[bad_scale] = 1.0

    X_scaled = X_tilde / scale
    n = max(len(y_tilde), 1)

    scores = np.abs(X_scaled.T @ y_tilde) / n
    alpha_max = float(np.max(scores)) if scores.size > 0 else 1e-12
    alpha_max = max(alpha_max, 1e-12)

    # For ElasticNet, all-zero threshold scales roughly as 1 / l1_ratio
    if model_kind == "elasticnet":
        alpha_max = alpha_max / max(l1_ratio, 1e-6)

    alpha_min = max(alpha_max * ratio, 1e-12)
    return np.geomspace(alpha_max, alpha_min, n_alphas)


def make_penalized_pipeline(
    alpha: float,
    model_kind: str,
    l1_ratio: float,
    positive_coefs: bool,
) -> Pipeline:
    if model_kind == "lasso":
        estimator = Lasso(
            alpha=alpha,
            positive=positive_coefs,
            fit_intercept=False,
            max_iter=1_000_000,
            tol=1e-4,
        )
    elif model_kind == "elasticnet":
        estimator = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            positive=positive_coefs,
            fit_intercept=False,
            max_iter=1_000_000,
            tol=1e-4,
        )
    else:
        raise ValueError("model_kind must be 'lasso' or 'elasticnet'.")

    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("model", estimator),
        ]
    )


def fit_absorbed_penalized(
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    alpha: float,
    n_groups: int,
    model_kind: str,
    l1_ratio: float,
    positive_coefs: bool,
) -> tuple[Pipeline, np.ndarray, np.ndarray, float]:
    """
    Fit lasso/elastic-net with absorbed fixed effects.

    Returns
    -------
    model : fitted sklearn pipeline on transformed data
    beta_orig : coefficients on original X scale
    group_intercepts : per-group intercepts
    fallback_intercept : used for unseen groups at prediction time
    """
    counts, y_means, X_means = fe_group_stats(X_train, y_train, g_train, n_groups)

    X_tilde = X_train - X_means[g_train]
    y_tilde = y_train - y_means[g_train]

    model = make_penalized_pipeline(
        alpha=alpha,
        model_kind=model_kind,
        l1_ratio=l1_ratio,
        positive_coefs=positive_coefs,
    )
    model.fit(X_tilde, y_tilde)

    scaler = model.named_steps["scaler"]
    penalized = model.named_steps["model"]

    scale = scaler.scale_.copy()
    bad_scale = ~np.isfinite(scale) | (scale <= 1e-12)
    scale[bad_scale] = 1.0
    beta_orig = penalized.coef_ / scale

    group_intercepts = np.full(n_groups, np.nan, dtype=float)
    seen = counts > 0
    if np.any(seen):
        group_intercepts[seen] = y_means[seen] - X_means[seen] @ beta_orig

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
    Predict in original scale:
        y_hat = alpha_g + X @ beta
    """
    xb = X @ beta_orig
    alpha = group_intercepts[g]

    unseen = ~np.isfinite(alpha)
    if np.any(unseen):
        alpha = alpha.copy()
        alpha[unseen] = fallback_intercept

    return xb + alpha


def count_nonzero_coefs(beta: np.ndarray, tol: float = 1e-12) -> int:
    return int(np.sum(np.abs(beta) > tol))


# =========================================================
# CROSS-VALIDATION
# =========================================================

def cv_alpha_stats(
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    n_groups: int,
    alphas: np.ndarray,
    model_kind: str,
    l1_ratio: float,
    positive_coefs: bool,
    cv_splits: int = 10,
    seed: int = 1702,
    verbose: bool = True,
    alpha_progress_step: int = 25,
    nonzero_tol: float = 1e-12,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Cross-validated MSE path.

    Returns
    -------
    alpha_min : alpha with best mean CV MSE
    alpha_1se : largest alpha within one SE of the minimum-MSE alpha
    mse_mean  : mean CV MSE for each alpha
    mse_se    : standard error of CV MSE for each alpha
    """
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    mse_path = np.zeros((len(alphas), cv_splits), dtype=float)

    log(
        f"[CV] Starting {cv_splits}-fold CV over {len(alphas)} alpha values "
        f"(model={model_kind}, l1_ratio={l1_ratio}, positive={positive_coefs})",
        verbose,
    )

    for fold_idx, (fit_idx, val_idx) in enumerate(cv.split(X_train), start=1):
        log(
            f"[CV] Fold {fold_idx}/{cv_splits} "
            f"(train_n={len(fit_idx)}, val_n={len(val_idx)})",
            verbose,
        )

        X_fit = X_train[fit_idx]
        y_fit = y_train[fit_idx]
        g_fit = g_train[fit_idx]

        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        g_val = g_train[val_idx]

        for alpha_idx, alpha in enumerate(alphas, start=1):
            _, beta_orig, group_intercepts, fallback_intercept = fit_absorbed_penalized(
                X_fit,
                y_fit,
                g_fit,
                alpha=float(alpha),
                n_groups=n_groups,
                model_kind=model_kind,
                l1_ratio=l1_ratio,
                positive_coefs=positive_coefs,
            )
            y_hat = predict_with_absorbed_fe(
                X_val, g_val, beta_orig, group_intercepts, fallback_intercept
            )
            mse_path[alpha_idx - 1, fold_idx - 1] = np.mean((y_val - y_hat) ** 2)

            should_print = (
                alpha_idx == 1
                or alpha_idx == len(alphas)
                or alpha_idx % max(alpha_progress_step, 1) == 0
            )
            if should_print:
                nz = count_nonzero_coefs(beta_orig, tol=nonzero_tol)
                log(
                    f"    alpha {alpha_idx:>3}/{len(alphas)} = {alpha:.6g} | "
                    f"nonzero={nz:>4} | fold MSE={mse_path[alpha_idx - 1, fold_idx - 1]:.6g}",
                    verbose,
                )

        fold_best = float(np.min(mse_path[:, fold_idx - 1]))
        log(f"[CV] Fold {fold_idx}/{cv_splits} done. Best fold MSE={fold_best:.6g}", verbose)

    mse_mean = mse_path.mean(axis=1)
    mse_se = mse_path.std(axis=1, ddof=1) / np.sqrt(cv_splits)

    min_idx = int(np.argmin(mse_mean))
    threshold = mse_mean[min_idx] + mse_se[min_idx]

    # Grid is descending (largest alpha first), so first qualifying alpha is the largest one.
    candidate_idx = np.where(mse_mean <= threshold)[0]
    one_se_idx = int(candidate_idx[0]) if len(candidate_idx) > 0 else min_idx

    alpha_min = float(alphas[min_idx])
    alpha_1se = float(alphas[one_se_idx])

    log(
        f"[CV] Done. alpha_min={alpha_min:.6g}, alpha_1se={alpha_1se:.6g}, "
        f"best mean MSE={mse_mean[min_idx]:.6g}",
        verbose,
    )

    return alpha_min, alpha_1se, mse_mean, mse_se


# =========================================================
# SUPPORT-AWARE FINAL ALPHA CHOICE
# =========================================================

def evaluate_support_path_on_full_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    n_groups: int,
    alphas: np.ndarray,
    model_kind: str,
    l1_ratio: float,
    positive_coefs: bool,
    nonzero_tol: float = 1e-12,
    verbose: bool = True,
    progress_step: int = 25,
) -> np.ndarray:
    """
    Fit the model on the FULL training sample for each alpha and record
    the number of nonzero coefficients.

    This is the key fix: it evaluates support over the whole alpha grid,
    not just by moving downward from one chosen alpha.
    """
    log("[Support] Evaluating support path on full training sample...", verbose)

    support_path = np.zeros(len(alphas), dtype=int)

    for i, alpha in enumerate(alphas, start=1):
        _, beta_orig, _, _ = fit_absorbed_penalized(
            X_train=X_train,
            y_train=y_train,
            g_train=g_train,
            alpha=float(alpha),
            n_groups=n_groups,
            model_kind=model_kind,
            l1_ratio=l1_ratio,
            positive_coefs=positive_coefs,
        )
        support_path[i - 1] = count_nonzero_coefs(beta_orig, tol=nonzero_tol)

        should_print = (
            i == 1
            or i == len(alphas)
            or i % max(progress_step, 1) == 0
        )
        if should_print:
            log(
                f"    alpha {i:>3}/{len(alphas)} = {alpha:.6g} | "
                f"full-train nonzero={support_path[i - 1]}",
                verbose,
            )

    max_support = int(np.max(support_path))
    max_idx = int(np.argmax(support_path))
    log(
        f"[Support] Done. Max full-train support = {max_support} at alpha={alphas[max_idx]:.6g}",
        verbose,
    )

    return support_path


def choose_base_alpha(
    alpha_min: float,
    alpha_1se: float,
    alpha_selection: str,
    alpha_scale: float,
) -> float:
    """
    Base choice if no support constraint is imposed.
    """
    if alpha_selection == "1se":
        final_alpha = alpha_1se
    elif alpha_selection == "min":
        final_alpha = alpha_min
    elif alpha_selection == "scaled_min":
        final_alpha = max(alpha_min * alpha_scale, 1e-12)
    else:
        raise ValueError("alpha_selection must be one of: '1se', 'min', 'scaled_min'")

    return float(final_alpha)


def choose_alpha_with_support_constraint(
    alphas: np.ndarray,
    mse_mean: np.ndarray,
    support_path: np.ndarray,
    alpha_min: float,
    alpha_1se: float,
    alpha_selection: str,
    alpha_scale: float,
    min_nonzero_coefs: int | None,
    verbose: bool = True,
) -> float:
    """
    Choose final alpha.

    Logic:
    - If MIN_NONZERO_COEFS is None:
        use the base rule (1se / min / scaled_min).
    - Else:
        choose the alpha WITHIN THE GRID that has the lowest CV MSE among those
        with support >= MIN_NONZERO_COEFS.
    - If none meet the target:
        choose the alpha with the largest support, breaking ties by lower CV MSE.

    This avoids the bad behavior of only searching toward smaller alpha.
    """
    base_alpha = choose_base_alpha(
        alpha_min=alpha_min,
        alpha_1se=alpha_1se,
        alpha_selection=alpha_selection,
        alpha_scale=alpha_scale,
    )

    if min_nonzero_coefs is None:
        log(
            f"[Alpha choice] MIN_NONZERO_COEFS is None. "
            f"Using base alpha={base_alpha:.6g} from rule '{alpha_selection}'.",
            verbose,
        )
        return base_alpha

    eligible = support_path >= min_nonzero_coefs

    if np.any(eligible):
        masked_mse = np.where(eligible, mse_mean, np.inf)
        idx = int(np.argmin(masked_mse))
        chosen_alpha = float(alphas[idx])

        log(
            f"[Alpha choice] Found {int(np.sum(eligible))} alpha values with "
            f"support >= {min_nonzero_coefs}.",
            verbose,
        )
        log(
            f"[Alpha choice] Choosing alpha={chosen_alpha:.6g} "
            f"(support={support_path[idx]}, mean CV MSE={mse_mean[idx]:.6g}).",
            verbose,
        )
        return chosen_alpha

    max_support = int(np.max(support_path))
    candidates = np.where(support_path == max_support)[0]
    idx = int(candidates[np.argmin(mse_mean[candidates])])
    chosen_alpha = float(alphas[idx])

    log(
        f"[Alpha choice] No alpha reaches support >= {min_nonzero_coefs}. "
        f"Best achievable support is {max_support}.",
        verbose,
    )
    log(
        f"[Alpha choice] Choosing alpha={chosen_alpha:.6g} among max-support alphas "
        f"(mean CV MSE={mse_mean[idx]:.6g}).",
        verbose,
    )
    return chosen_alpha


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]

    # =========================
    # USER SETTINGS
    # =========================

    # Main data file
    data_path = project_dir / "data" / "temp_wfd_1702_6_w_certainty_quartiles_for_elastic.dta"

    # Subset .dta containing allowed suffix numbers
    subset_path = Path(
        "D:/cursor_projects/vc_cc_positions_cloud_site/output/list_of_sm_related_k1000_gbrl.dta"
    )

    # Column in subset_path that contains the allowed suffixes; None = auto-detect
    subset_col = None

    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    yvar = "w_xsalesmktrawCIQ"
    fevar = "bucket16"

    # Predictor family
    # x_prefix = "w_salary_k1000_"
    x_prefix = "w_total_compensation_k1000_"
    x_min = 1
    x_max = 1004

    # =========================
    # MODEL SETTINGS
    # =========================

    MODEL_KIND = "elasticnet"   # "elasticnet" or "lasso"
    L1_RATIO = 0.7              # used only for elasticnet
    POSITIVE_COEFS = True       # keep nonnegativity restriction

    N_ALPHAS = 400
    ALPHA_GRID_RATIO = 1e-10

    # If MIN_NONZERO_COEFS is None, this base rule is used directly.
    # If MIN_NONZERO_COEFS is not None, the support constraint takes priority.
    ALPHA_SELECTION = "scaled_min"   # "1se", "min", or "scaled_min"
    ALPHA_SCALE = 0.01               # used only when ALPHA_SELECTION == "scaled_min"

    # This now works over the FULL alpha grid, not just by pushing alpha downward.
    MIN_NONZERO_COEFS = 50           # set None to disable support floor

    NONZERO_TOL = 1e-12

    CV_SPLITS = 10
    RANDOM_SEED = 1702

    # =========================
    # PROGRESS PRINT SETTINGS
    # =========================

    VERBOSE = True
    ALPHA_PROGRESS_STEP = 25
    SUPPORT_PROGRESS_STEP = 25

    # =========================
    # LOAD RESTRICTED PREDICTOR SET
    # =========================

    log("[1/8] Checking files and loading allowed suffixes...", VERBOSE)

    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    allowed_suffixes = load_allowed_suffixes(
        subset_path=subset_path,
        subset_col=subset_col,
        min_suffix=x_min,
        max_suffix=x_max,
    )
    xvars = [f"{x_prefix}{i}" for i in allowed_suffixes]

    log(f"    Loaded {len(allowed_suffixes)} allowed suffixes from: {subset_path}", VERBOSE)
    log(f"    Using restricted predictor set: {len(xvars)} variables", VERBOSE)

    # =========================
    # LOAD DATA
    # =========================

    log("[2/8] Loading main dataset...", VERBOSE)
    df = pd.read_stata(data_path).copy()
    log(f"    Main dataset shape: {df.shape[0]} rows x {df.shape[1]} cols", VERBOSE)

    required_cols = [yvar, fevar] + xvars
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            "These required columns are missing from the main data file:\n"
            + "\n".join(missing_cols)
        )

    # Match your original logic: split only complete cases for y + FE + restricted xvars
    complete_mask = df[required_cols].notna().all(axis=1)
    complete_idx = df.index[complete_mask].to_numpy()

    if len(complete_idx) == 0:
        raise ValueError("No complete-case observations remain after applying y + FE + restricted xvars.")

    rng = np.random.default_rng(RANDOM_SEED)
    n_complete = len(complete_idx)
    n_train = int(round(0.8 * n_complete))
    train_idx = rng.choice(complete_idx, size=n_train, replace=False)

    split = pd.Series(pd.NA, index=df.index, dtype="Int64")
    split.loc[complete_idx] = 2
    split.loc[train_idx] = 1
    df = df.assign(split=split)

    # Integer-code FE on the complete-case sample only
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

    log(
        f"    Complete-case rows: {n_complete} | train: {len(y_train)} | "
        f"test: {len(y_test)} | groups: {n_groups}",
        VERBOSE,
    )

    # =========================
    # BUILD ALPHA GRID
    # =========================

    log("[3/8] Building alpha grid...", VERBOSE)
    alphas = build_alpha_grid(
        X_train=X_train,
        y_train=y_train,
        g_train=g_train,
        n_groups=n_groups,
        model_kind=MODEL_KIND,
        l1_ratio=L1_RATIO,
        n_alphas=N_ALPHAS,
        ratio=ALPHA_GRID_RATIO,
    )
    log(
        f"    Alpha grid built: {len(alphas)} values | "
        f"max={alphas[0]:.6g} | min={alphas[-1]:.6g}",
        VERBOSE,
    )

    # =========================
    # CROSS-VALIDATION
    # =========================

    log("[4/8] Running cross-validation...", VERBOSE)
    alpha_min, alpha_1se, mse_mean, mse_se = cv_alpha_stats(
        X_train=X_train,
        y_train=y_train,
        g_train=g_train,
        n_groups=n_groups,
        alphas=alphas,
        model_kind=MODEL_KIND,
        l1_ratio=L1_RATIO,
        positive_coefs=POSITIVE_COEFS,
        cv_splits=CV_SPLITS,
        seed=RANDOM_SEED,
        verbose=VERBOSE,
        alpha_progress_step=ALPHA_PROGRESS_STEP,
        nonzero_tol=NONZERO_TOL,
    )

    # =========================
    # FULL-TRAIN SUPPORT PATH
    # =========================

    log("[5/8] Evaluating full-train support path...", VERBOSE)
    support_path = evaluate_support_path_on_full_train(
        X_train=X_train,
        y_train=y_train,
        g_train=g_train,
        n_groups=n_groups,
        alphas=alphas,
        model_kind=MODEL_KIND,
        l1_ratio=L1_RATIO,
        positive_coefs=POSITIVE_COEFS,
        nonzero_tol=NONZERO_TOL,
        verbose=VERBOSE,
        progress_step=SUPPORT_PROGRESS_STEP,
    )

    # Optional: print a quick summary near the dense region
    max_support = int(np.max(support_path))
    max_support_idx = int(np.argmax(support_path))
    log(
        f"    Full-train max support = {max_support} at alpha={alphas[max_support_idx]:.6g}",
        VERBOSE,
    )

    # =========================
    # CHOOSE FINAL ALPHA
    # =========================

    log("[6/8] Choosing final alpha...", VERBOSE)
    base_alpha = choose_base_alpha(
        alpha_min=alpha_min,
        alpha_1se=alpha_1se,
        alpha_selection=ALPHA_SELECTION,
        alpha_scale=ALPHA_SCALE,
    )
    log(
        f"    Base alpha from rule '{ALPHA_SELECTION}': {base_alpha:.6g}",
        VERBOSE,
    )

    final_alpha = choose_alpha_with_support_constraint(
        alphas=alphas,
        mse_mean=mse_mean,
        support_path=support_path,
        alpha_min=alpha_min,
        alpha_1se=alpha_1se,
        alpha_selection=ALPHA_SELECTION,
        alpha_scale=ALPHA_SCALE,
        min_nonzero_coefs=MIN_NONZERO_COEFS,
        verbose=VERBOSE,
    )
    log(f"    Final alpha used: {final_alpha:.6g}", VERBOSE)

    # =========================
    # FINAL FIT
    # =========================

    log("[7/8] Fitting final model...", VERBOSE)
    _, coef_orig, group_intercepts, fallback_intercept = fit_absorbed_penalized(
        X_train=X_train,
        y_train=y_train,
        g_train=g_train,
        alpha=final_alpha,
        n_groups=n_groups,
        model_kind=MODEL_KIND,
        l1_ratio=L1_RATIO,
        positive_coefs=POSITIVE_COEFS,
    )

    selected_mask = np.abs(coef_orig) > NONZERO_TOL
    selected_vars = [v for v, keep in zip(xvars, selected_mask) if keep]
    selected_weights = coef_orig[selected_mask]

    selected_df = pd.DataFrame({"variable": selected_vars, "weight": selected_weights})
    selected_df = selected_df.sort_values("variable").reset_index(drop=True)

    final_nz = len(selected_vars)
    log(f"    Final selected variable count: {final_nz}", VERBOSE)

    # =========================
    # SAVE OUTPUTS
    # =========================

    log("[8/8] Saving outputs...", VERBOSE)

    selected_path = output_dir / "elasticnet_wfd_1702_6_compensation_selected_weights_gbrl_fe_supportaware.csv"
    selected_df.to_csv(selected_path, index=False)

    yhat_test = predict_with_absorbed_fe(
        X_test, g_test, coef_orig, group_intercepts, fallback_intercept
    )

    pred_df = df.loc[test_mask, [yvar, fevar]].copy()
    pred_df["yhat"] = yhat_test
    pred_df["err"] = pred_df[yvar] - pred_df["yhat"]
    pred_df["abs_err"] = pred_df["err"].abs()
    pred_df["sq_err"] = pred_df["err"] ** 2

    pred_path = output_dir / "elasticnet_wfd_1702_6_compensation_test_predictions_gbrl_fe_supportaware.csv"
    pred_df.to_csv(pred_path, index=False)

    n_test = len(pred_df)
    rmse = float(np.sqrt(pred_df["sq_err"].mean())) if n_test > 0 else np.nan
    mae = float(pred_df["abs_err"].mean()) if n_test > 0 else np.nan

    if n_test > 1 and pred_df[yvar].std(ddof=1) > 0 and pred_df["yhat"].std(ddof=1) > 0:
        r2_holdout = float(np.corrcoef(pred_df[yvar], pred_df["yhat"])[0, 1] ** 2)
    else:
        r2_holdout = np.nan

    results_path = output_dir / "elasticnet_wfd_1702_6_compensation_results_gbrl_fe_supportaware.txt"
    with results_path.open("w", encoding="utf-8") as fout:
        fout.write("Restricted absorbed penalized regression run (support-aware alpha choice)\n")
        fout.write(f"Dependent variable: {yvar}\n")
        fout.write(f"Fixed effects (absorbed, unpenalized): i.{fevar}\n")
        fout.write(f"Main data file: {data_path}\n")
        fout.write(f"Subset file: {subset_path}\n")
        fout.write(f"Allowed suffix count: {len(allowed_suffixes)}\n")
        fout.write(f"Predictor family: {x_prefix}{x_min}-{x_prefix}{x_max}\n")
        fout.write(f"Restricted predictor count: {len(xvars)}\n\n")

        fout.write("Model settings:\n")
        fout.write(f"MODEL_KIND         = {MODEL_KIND}\n")
        fout.write(f"L1_RATIO           = {L1_RATIO:.6g}\n")
        fout.write(f"POSITIVE_COEFS     = {POSITIVE_COEFS}\n")
        fout.write(f"N_ALPHAS           = {N_ALPHAS}\n")
        fout.write(f"ALPHA_GRID_RATIO   = {ALPHA_GRID_RATIO:.6g}\n")
        fout.write(f"ALPHA_SELECTION    = {ALPHA_SELECTION}\n")
        fout.write(f"ALPHA_SCALE        = {ALPHA_SCALE:.6g}\n")
        fout.write(f"MIN_NONZERO_COEFS  = {MIN_NONZERO_COEFS}\n")
        fout.write(f"Train/test split seed = {RANDOM_SEED}\n")
        fout.write(f"CV splits             = {CV_SPLITS}\n\n")

        fout.write("Alpha choices:\n")
        fout.write(f"CV alpha(min-MSE)     = {alpha_min:.10g}\n")
        fout.write(f"CV alpha(one-SE)      = {alpha_1se:.10g}\n")
        fout.write(f"Base alpha            = {base_alpha:.10g}\n")
        fout.write(f"Final alpha used      = {final_alpha:.10g}\n\n")

        fout.write("Support path summary:\n")
        fout.write(f"Max full-train support = {int(np.max(support_path))}\n")
        fout.write(
            f"Alpha at max support   = {float(alphas[int(np.argmax(support_path))]):.10g}\n\n"
        )

        fout.write("Allowed suffixes:\n")
        fout.write(" ".join(map(str, allowed_suffixes)) + "\n\n")

        fout.write("Selected variables:\n")
        if selected_vars:
            fout.write(" ".join(selected_vars) + "\n\n")
        else:
            fout.write("(none)\n\n")

        fout.write(f"Selected variable count = {len(selected_vars)}\n\n")

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

    log("Done.", VERBOSE)
    print(f"Alpha(min-MSE): {alpha_min:.10g}")
    print(f"Alpha(1-SE):    {alpha_1se:.10g}")
    print(f"Base alpha:     {base_alpha:.10g}")
    print(f"Final alpha:    {final_alpha:.10g}")
    print(f"Selected vars:  {len(selected_vars)}")
    print(f"Saved: {selected_path}")
    print(f"Saved: {pred_path}")
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()