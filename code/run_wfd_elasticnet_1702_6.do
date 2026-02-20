version 18.0

* Reproducible elastic-net workflow for temp_wfd_1702_6.dta
* Goal: predict w_xsalesmktrawCIQ from w_salary_k50_1-w_salary_k50_53

* Project root path (edit if moved to a different machine/folder)
local project_dir "D:/cursor_projects/vc_cc_positions_cloud_site"

capture confirm file "`project_dir'/data/temp_wfd_1702_6.dta"
if _rc {
    di as error "Cannot find data file: `project_dir'/data/temp_wfd_1702_6.dta"
    exit 601
}

use "`project_dir'/data/temp_wfd_1702_6.dta", clear

local yvar  w_xsalesmktrawCIQ
local xvars w_salary_k50_1-w_salary_k50_53

* 1) Train/test split (80/20) on complete cases for y and xvars
set seed 1702
capture drop split yhat err abs_err sq_err
splitsample `yvar' `xvars', generate(split) split(.8 .2)
tab split

* 2) Fit elastic net (lasso-like setting)
* alpha(1): pure lasso penalty
* serule: one-standard-error rule (usually more stable/sparser)
* gridminok + wide grid: avoids r(430) when CV minimum is at boundary
elasticnet linear `yvar' `xvars' if split==1, ///
    alpha(1) selection(cv, serule gridminok) grid(200, ratio(1e-6))

local selected_vars `e(allvars_sel)'
display as text "Selected variables:"
display as result "`selected_vars'"

* Store selected variables and their penalized coefficients (weights)
local n_sel : word count `selected_vars'
preserve
clear
set obs `n_sel'
gen str64 variable = ""
gen double weight = .
local i = 1
foreach v of local selected_vars {
    replace variable = "`v'" in `i'
    replace weight   = _b[`v'] in `i'
    local ++i
}
sort variable
save "`project_dir'/output/elasticnet_wfd_1702_6_selected_weights.dta", replace
export delimited using "`project_dir'/output/elasticnet_wfd_1702_6_selected_weights.csv", replace
restore

* 3) Holdout (test sample) evaluation
predict double yhat if split==2
gen double err    = `yvar' - yhat if split==2
gen double abs_err = abs(err) if split==2
gen double sq_err  = err^2 if split==2

quietly count if split==2
scalar n_test = r(N)

quietly summarize sq_err if split==2, meanonly
scalar rmse = sqrt(r(mean))

quietly summarize abs_err if split==2, meanonly
scalar mae = r(mean)

quietly correlate `yvar' yhat if split==2
scalar r2_holdout = r(rho)^2

display as text "Holdout metrics (split==2):"
display as result "N test = " %10.0f n_test
display as result "RMSE   = " %12.4f rmse
display as result "MAE    = " %12.4f mae
display as result "R2     = " %12.4f r2_holdout

* 4) Save holdout predictions for diagnostics
preserve
keep if split==2
keep `yvar' yhat err abs_err sq_err
save "`project_dir'/output/elasticnet_wfd_1702_6_test_predictions.dta", replace
restore

* 5) Write compact results report
file open fout using "`project_dir'/output/elasticnet_wfd_1702_6_results.txt", write replace
file write fout "Elastic net run on temp_wfd_1702_6.dta" _n
file write fout "Dependent variable: `yvar'" _n
file write fout "Predictors: `xvars'" _n
file write fout "Train/test split seed: 1702" _n
file write fout "Specification: alpha(1) selection(cv, serule gridminok) grid(200, ratio(1e-6))" _n _n
file write fout "Selected variables:" _n
file write fout "`selected_vars'" _n _n
file write fout "Selected variables and weights:" _n
foreach v of local selected_vars {
    file write fout "`v'" _tab %16.8f (_b[`v']) _n
}
file write fout _n
file write fout "Holdout metrics (split==2):" _n
file write fout "N test = " %10.0f (n_test) _n
file write fout "RMSE   = " %12.4f (rmse) _n
file write fout "MAE    = " %12.4f (mae) _n
file write fout "R2     = " %12.4f (r2_holdout) _n
file close fout

display as text "Saved: `project_dir'/data/elasticnet_wfd_1702_6_test_predictions.dta"
display as text "Saved: `project_dir'/code/elasticnet_wfd_1702_6_results.txt"
display as text "Saved: `project_dir'/code/elasticnet_wfd_1702_6_selected_weights.dta"
display as text "Saved: `project_dir'/code/elasticnet_wfd_1702_6_selected_weights.csv"
