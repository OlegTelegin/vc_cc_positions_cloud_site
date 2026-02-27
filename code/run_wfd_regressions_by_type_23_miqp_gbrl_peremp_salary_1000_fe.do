clear all
set more off

local project_dir "D:/cursor_projects/vc_cc_positions_cloud_site"

local sm_type "selected_vars_gbrl_peremp_salary_1000"
local rhs_salary_var "w_sal_k10_8_sel_ps50"
local rhs_weight_var "w_wgt_k10_8_sel_ps50"
local rhs_total_comp_var "w_tcmp_k10_8_sel_ps50"
local rhs_per_salary_var "w_psal_k10_8_sel_ps50"
local rhs_per_weight_var "w_pwgt_k10_8_sel_ps50"
local rhs_per_total_comp_var "w_ptcmp_k10_8_sel_ps50"

* Read selected variables list (no weights)
import delimited "`project_dir'/data/miqp_gbrl_1000_peremp_salary_fe.csv", clear varnames(1)

capture confirm variable selected_var
if _rc {
    di as error "Selected-vars file is missing column: selected_var"
    exit 111
}

drop if missing(selected_var)
quietly count
if r(N) == 0 {
    di as error "No selected variables found in `project_dir'/data/miqp_gbrl_1000_peremp_salary_fe.csv"
    exit 2000
}

levelsof selected_var, local(selected_vars)
local selected_ids ""
foreach v of local selected_vars {
    if !regexm("`v'", "^[a-z_]+_k1000_([0-9]+)$") {
        di as error "Unexpected selected_var format: `v' (expected *_k1000_#)"
        exit 111
    }
    local id = regexs(1)
    local selected_ids `selected_ids' `id'
}

tempfile results_tmp
tempname posth

postfile `posth' ///
    int regression_number ///
    str40 sm_type ///
    double coefficient ///
    double r2 ///
    using `results_tmp', replace

* -------- First sample (full data) --------
use "`project_dir'/data/temp_wfd_1702_5_w_certainty_quartiles.dta", clear

gen double rhs_salary_sum = 0
gen double rhs_weight_sum = 0
gen double rhs_total_comp_sum = 0
foreach id of local selected_ids {
    local v_salary salary_k1000_`id'
    local v_weight weight_k1000_`id'
    local v_total_comp total_compensation_k1000_`id'

    capture confirm variable `v_salary'
    if _rc {
        di as error "Variable `v_salary' not found in temp_wfd_1702_5.dta"
        exit 111
    }
    capture confirm variable `v_weight'
    if _rc {
        di as error "Variable `v_weight' not found in temp_wfd_1702_5.dta"
        exit 111
    }
    capture confirm variable `v_total_comp'
    if _rc {
        di as error "Variable `v_total_comp' not found in temp_wfd_1702_5.dta"
        exit 111
    }

    replace rhs_salary_sum = rhs_salary_sum + `v_salary'
    replace rhs_weight_sum = rhs_weight_sum + `v_weight'
    replace rhs_total_comp_sum = rhs_total_comp_sum + `v_total_comp'
}

gen t = rhs_salary_sum
replace t = . if t == 0
winsor t, gen(`rhs_salary_var') p(0.01)
drop t

gen t = rhs_weight_sum
replace t = . if t == 0
winsor t, gen(`rhs_weight_var') p(0.01)
drop t

gen t = rhs_total_comp_sum
replace t = . if t == 0
winsor t, gen(`rhs_total_comp_var') p(0.01)
drop t

gen t = rhs_salary_sum/total_wage_bill
replace t = . if t == 0
winsor t, gen(`rhs_per_salary_var') p(0.01)
drop t

gen t = rhs_weight_sum/total_employment
replace t = . if t == 0
winsor t, gen(`rhs_per_weight_var') p(0.01)
drop t

gen t = rhs_total_comp_sum/total_compensation_bill
replace t = . if t == 0
winsor t, gen(`rhs_per_total_comp_var') p(0.01)
drop t

sum total_employment if xsalesmktrawCIQ != ., detail
local perc = r(p25)

* 1
regress w_xsalesmktrawCIQ `rhs_salary_var' i.bucket16, r
post `posth' (1) ("`sm_type'") (_b[`rhs_salary_var']) (e(r2))

* save estimation sample
gen byte samp1 = e(sample)

* 2) Get fitted value from model 1
predict double xb1 if samp1, xb

* 3) Keep only the bucket16 contribution (NOT the constant, NOT salary)
gen double bucket_fix = xb1 - _b[_cons] - _b[`rhs_salary_var']*`rhs_salary_var' if samp1

* 4) Second regression, with bucket contribution fixed at coefficient 1
constraint drop _all
constraint 1 bucket_fix = 1

* 2
cnsreg w_xsalesmktrawCIQ `rhs_total_comp_var' bucket_fix if samp1, constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (2) ("`sm_type'") (_b[`rhs_total_comp_var']) (R2_c)

* 5
cnsreg w_xsalesmktrawCIQ `rhs_weight_var' bucket_fix if samp1, constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (5) ("`sm_type'") (_b[`rhs_weight_var']) (R2_c)

* 13
cnsreg w_xsalesmktrawCIQ `rhs_salary_var' bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (13) ("`sm_type'") (_b[`rhs_salary_var']) (R2_c)

* 14
cnsreg w_xsalesmktrawCIQ `rhs_total_comp_var' bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (14) ("`sm_type'") (_b[`rhs_total_comp_var']) (R2_c)

* 17
cnsreg w_xsalesmktrawCIQ `rhs_weight_var' bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (17) ("`sm_type'") (_b[`rhs_weight_var']) (R2_c)

* 3
regress w_per_xsalesmktrawCIQ `rhs_per_salary_var' i.bucket16, r
post `posth' (3) ("`sm_type'") (_b[`rhs_per_salary_var']) (e(r2))

drop samp1 bucket_fix xb1
* save estimation sample
gen byte samp1 = e(sample)

* 2) Get fitted value from model 1
predict double xb1 if samp1, xb

* 3) Keep only the bucket16 contribution (NOT the constant, NOT salary)
gen double bucket_fix = xb1 - _b[_cons] - _b[`rhs_per_salary_var']*`rhs_per_salary_var' if samp1

* 4) Second regression, with bucket contribution fixed at coefficient 1
constraint drop _all
constraint 1 bucket_fix = 1

* 4
cnsreg w_per_xsalesmktrawCIQ `rhs_per_total_comp_var' bucket_fix if samp1, constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_per_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_per_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (4) ("`sm_type'") (_b[`rhs_per_total_comp_var']) (R2_c)

* 6
cnsreg w_per_xsalesmktrawCIQ `rhs_per_weight_var' bucket_fix if samp1, constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_per_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_per_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (6) ("`sm_type'") (_b[`rhs_per_weight_var']) (R2_c)

* 15
cnsreg w_per_xsalesmktrawCIQ `rhs_per_salary_var' bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_per_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_per_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (15) ("`sm_type'") (_b[`rhs_per_salary_var']) (R2_c)

* 16
cnsreg w_per_xsalesmktrawCIQ `rhs_per_total_comp_var' bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_per_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_per_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (16) ("`sm_type'") (_b[`rhs_per_total_comp_var']) (R2_c)

* 18
cnsreg w_per_xsalesmktrawCIQ `rhs_per_weight_var' bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_per_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_per_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (18) ("`sm_type'") (_b[`rhs_per_weight_var']) (R2_c)

* -------- Second sample (filtered on emp_cstat2wfd_2) --------
use "`project_dir'/data/temp_wfd_1702_5_w_certainty_quartiles.dta", clear

gen double rhs_salary_sum = 0
gen double rhs_weight_sum = 0
gen double rhs_total_comp_sum = 0
foreach id of local selected_ids {
    local v_salary salary_k1000_`id'
    local v_weight weight_k1000_`id'
    local v_total_comp total_compensation_k1000_`id'

    capture confirm variable `v_salary'
    if _rc {
        di as error "Variable `v_salary' not found in temp_wfd_1702_5.dta"
        exit 111
    }
    capture confirm variable `v_weight'
    if _rc {
        di as error "Variable `v_weight' not found in temp_wfd_1702_5.dta"
        exit 111
    }
    capture confirm variable `v_total_comp'
    if _rc {
        di as error "Variable `v_total_comp' not found in temp_wfd_1702_5.dta"
        exit 111
    }

    replace rhs_salary_sum = rhs_salary_sum + `v_salary'
    replace rhs_weight_sum = rhs_weight_sum + `v_weight'
    replace rhs_total_comp_sum = rhs_total_comp_sum + `v_total_comp'
}

gen t = rhs_salary_sum
replace t = . if t == 0
winsor t, gen(`rhs_salary_var') p(0.01)
drop t

gen t = rhs_weight_sum
replace t = . if t == 0
winsor t, gen(`rhs_weight_var') p(0.01)
drop t

gen t = rhs_total_comp_sum
replace t = . if t == 0
winsor t, gen(`rhs_total_comp_var') p(0.01)
drop t

gen t = rhs_salary_sum/total_wage_bill
replace t = . if t == 0
winsor t, gen(`rhs_per_salary_var') p(0.01)
drop t

gen t = rhs_weight_sum/total_employment
replace t = . if t == 0
winsor t, gen(`rhs_per_weight_var') p(0.01)
drop t

gen t = rhs_total_comp_sum/total_compensation_bill
replace t = . if t == 0
winsor t, gen(`rhs_per_total_comp_var') p(0.01)
drop t

gen emp_cstat2wfd_2 = emp/total_employment
drop if emp_cstat2wfd_2 <= 0.25
drop if emp_cstat2wfd_2 > 2
drop emp_cstat2wfd_2

sum total_employment if xsalesmktrawCIQ != ., detail
local perc = r(p25)

* 7
regress w_xsalesmktrawCIQ `rhs_salary_var' i.bucket16, r
post `posth' (7) ("`sm_type'") (_b[`rhs_salary_var']) (e(r2))

* save estimation sample
gen byte samp1 = e(sample)

* 2) Get fitted value from model 1
predict double xb1 if samp1, xb

* 3) Keep only the bucket16 contribution (NOT the constant, NOT salary)
gen double bucket_fix = xb1 - _b[_cons] - _b[`rhs_salary_var']*`rhs_salary_var' if samp1

* 4) Second regression, with bucket contribution fixed at coefficient 1
constraint drop _all
constraint 1 bucket_fix = 1

* 8
cnsreg w_xsalesmktrawCIQ `rhs_total_comp_var' bucket_fix if samp1, constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (8) ("`sm_type'") (_b[`rhs_total_comp_var']) (R2_c)

* 11
cnsreg w_xsalesmktrawCIQ `rhs_weight_var' bucket_fix if samp1, constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (11) ("`sm_type'") (_b[`rhs_weight_var']) (R2_c)

* 19
cnsreg w_xsalesmktrawCIQ `rhs_salary_var' bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (19) ("`sm_type'") (_b[`rhs_salary_var']) (R2_c)

* 20
cnsreg w_xsalesmktrawCIQ `rhs_total_comp_var' bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (20) ("`sm_type'") (_b[`rhs_total_comp_var']) (R2_c)

* 23
cnsreg w_xsalesmktrawCIQ `rhs_weight_var' bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (23) ("`sm_type'") (_b[`rhs_weight_var']) (R2_c)

* 9
regress w_per_xsalesmktrawCIQ `rhs_per_salary_var' i.bucket16, r
post `posth' (9) ("`sm_type'") (_b[`rhs_per_salary_var']) (e(r2))

drop samp1 bucket_fix xb1
* save estimation sample
gen byte samp1 = e(sample)

* 2) Get fitted value from model 1
predict double xb1 if samp1, xb

* 3) Keep only the bucket16 contribution (NOT the constant, NOT salary)
gen double bucket_fix = xb1 - _b[_cons] - _b[`rhs_per_salary_var']*`rhs_per_salary_var' if samp1

* 4) Second regression, with bucket contribution fixed at coefficient 1
constraint drop _all
constraint 1 bucket_fix = 1

* 10
cnsreg w_per_xsalesmktrawCIQ `rhs_per_total_comp_var' bucket_fix if samp1, constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_per_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_per_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (10) ("`sm_type'") (_b[`rhs_per_total_comp_var']) (R2_c)

* 12
cnsreg w_per_xsalesmktrawCIQ `rhs_per_weight_var' bucket_fix if samp1, constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_per_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_per_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (12) ("`sm_type'") (_b[`rhs_per_weight_var']) (R2_c)

* 21
cnsreg w_per_xsalesmktrawCIQ `rhs_per_salary_var' bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_per_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_per_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (21) ("`sm_type'") (_b[`rhs_per_salary_var']) (R2_c)

* 22
cnsreg w_per_xsalesmktrawCIQ `rhs_per_total_comp_var' bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_per_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_per_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (22) ("`sm_type'") (_b[`rhs_per_total_comp_var']) (R2_c)

* 24
cnsreg w_per_xsalesmktrawCIQ `rhs_per_weight_var' bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

tempvar u u2 dev2
predict double `u' if e(sample), residuals
quietly summarize w_per_xsalesmktrawCIQ if e(sample), meanonly
scalar ybar = r(mean)

gen double `u2'  = `u'^2 if e(sample)
gen double `dev2' = (w_per_xsalesmktrawCIQ - ybar)^2 if e(sample)

quietly summarize `u2' if e(sample), meanonly
scalar RSS = r(sum)
quietly summarize `dev2' if e(sample), meanonly
scalar TSS = r(sum)

scalar R2_c = 1 - RSS/TSS
post `posth' (24) ("`sm_type'") (_b[`rhs_per_weight_var']) (R2_c)

postclose `posth'

use `results_tmp', clear
sort regression_number
save "..\output\wfd_regressions_by_type_23_miqp_gbrl_peremp_salary_1000_fe.dta", replace
