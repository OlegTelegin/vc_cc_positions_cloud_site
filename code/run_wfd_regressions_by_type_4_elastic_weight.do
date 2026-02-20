clear all
set more off

local sm_type "out_of_50_classes_elastic_weight"
local elastic_salary_var "w_salary_k10_8_elastic_w"
local elastic_weight_var "w_weight_k10_8_elastic_w"
local elastic_total_comp_var "w_tcomp_k10_8_elastic_w"
local elastic_per_salary_var "w_per_salary_k10_8_elastic_w"
local elastic_per_weight_var "w_per_weight_k10_8_elastic_w"
local elastic_per_total_comp_var "w_per_tcomp_k10_8_elastic_w"

* Read elastic-net selected variables and weights from weight-based model
import delimited "..\output\elasticnet_wfd_1702_6_weight_selected_weights.csv", clear varnames(1)

capture confirm variable variable
if _rc {
    di as error "Weights file is missing column: variable"
    exit 111
}
capture confirm variable weight
if _rc {
    di as error "Weights file is missing column: weight"
    exit 111
}

drop if missing(variable) | missing(weight)
quietly count
if r(N) == 0 {
    di as error "No elastic-net weights found in ..\output\elasticnet_wfd_1702_6_weight_selected_weights.csv"
    exit 2000
}

levelsof variable, local(elastic_vars)
local elastic_ids ""
foreach v of local elastic_vars {
    if !regexm("`v'", "^w_weight_k50_([0-9]+)$") {
        di as error "Unexpected variable name in weights file: `v' (expected w_weight_k50_#)"
        exit 111
    }
    local id = regexs(1)
    quietly summarize weight if variable == "`v'", meanonly
    local wt_`id' = r(mean)
    local elastic_ids `elastic_ids' `id'
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
use "..\data\temp_wfd_1702_5.dta", clear

gen double elastic_salary_sum = 0
gen double elastic_weight_sum = 0
gen double elastic_total_comp_sum = 0
foreach id of local elastic_ids {
    local v_salary salary_k50_`id'
    local v_weight weight_k50_`id'
    local v_total_comp total_compensation_k50_`id'

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

    replace elastic_salary_sum = elastic_salary_sum + (`wt_`id'') * `v_salary'
    replace elastic_weight_sum = elastic_weight_sum + (`wt_`id'') * `v_weight'
    replace elastic_total_comp_sum = elastic_total_comp_sum + (`wt_`id'') * `v_total_comp'
}

gen t = elastic_salary_sum
replace t = . if t == 0
winsor t, gen(`elastic_salary_var') p(0.01)
drop t

gen t = elastic_weight_sum
replace t = . if t == 0
winsor t, gen(`elastic_weight_var') p(0.01)
drop t

gen t = elastic_total_comp_sum
replace t = . if t == 0
winsor t, gen(`elastic_total_comp_var') p(0.01)
drop t

gen t = elastic_salary_sum/total_wage_bill
replace t = . if t == 0
winsor t, gen(`elastic_per_salary_var') p(0.01)
drop t

gen t = elastic_weight_sum/total_employment
replace t = . if t == 0
winsor t, gen(`elastic_per_weight_var') p(0.01)
drop t

gen t = elastic_total_comp_sum/total_compensation_bill
replace t = . if t == 0
winsor t, gen(`elastic_per_total_comp_var') p(0.01)
drop t

sum total_employment if xsalesmktrawCIQ != ., detail
local perc = r(p25)

* 1-6
regress w_xsalesmktrawCIQ `elastic_salary_var', r
post `posth' (1) ("`sm_type'") (_b[`elastic_salary_var']) (e(r2))

regress w_xsalesmktrawCIQ `elastic_total_comp_var', r
post `posth' (2) ("`sm_type'") (_b[`elastic_total_comp_var']) (e(r2))

regress w_per_xsalesmktrawCIQ `elastic_per_salary_var', r
post `posth' (3) ("`sm_type'") (_b[`elastic_per_salary_var']) (e(r2))

regress w_per_xsalesmktrawCIQ `elastic_per_total_comp_var', r
post `posth' (4) ("`sm_type'") (_b[`elastic_per_total_comp_var']) (e(r2))

regress w_xsalesmktrawCIQ `elastic_weight_var', r
post `posth' (5) ("`sm_type'") (_b[`elastic_weight_var']) (e(r2))

regress w_per_xsalesmktrawCIQ `elastic_per_weight_var', r
post `posth' (6) ("`sm_type'") (_b[`elastic_per_weight_var']) (e(r2))

* 13-18
regress w_xsalesmktrawCIQ `elastic_salary_var' if total_employment < `perc', r
post `posth' (13) ("`sm_type'") (_b[`elastic_salary_var']) (e(r2))

regress w_xsalesmktrawCIQ `elastic_total_comp_var' if total_employment < `perc', r
post `posth' (14) ("`sm_type'") (_b[`elastic_total_comp_var']) (e(r2))

regress w_per_xsalesmktrawCIQ `elastic_per_salary_var' if total_employment < `perc', r
post `posth' (15) ("`sm_type'") (_b[`elastic_per_salary_var']) (e(r2))

regress w_per_xsalesmktrawCIQ `elastic_per_total_comp_var' if total_employment < `perc', r
post `posth' (16) ("`sm_type'") (_b[`elastic_per_total_comp_var']) (e(r2))

regress w_xsalesmktrawCIQ `elastic_weight_var' if total_employment < `perc', r
post `posth' (17) ("`sm_type'") (_b[`elastic_weight_var']) (e(r2))

regress w_per_xsalesmktrawCIQ `elastic_per_weight_var' if total_employment < `perc', r
post `posth' (18) ("`sm_type'") (_b[`elastic_per_weight_var']) (e(r2))

* -------- Second sample (filtered on emp_cstat2wfd_2) --------
use "..\data\temp_wfd_1702_5.dta", clear

gen double elastic_salary_sum = 0
gen double elastic_weight_sum = 0
gen double elastic_total_comp_sum = 0
foreach id of local elastic_ids {
    local v_salary salary_k50_`id'
    local v_weight weight_k50_`id'
    local v_total_comp total_compensation_k50_`id'

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

    replace elastic_salary_sum = elastic_salary_sum + (`wt_`id'') * `v_salary'
    replace elastic_weight_sum = elastic_weight_sum + (`wt_`id'') * `v_weight'
    replace elastic_total_comp_sum = elastic_total_comp_sum + (`wt_`id'') * `v_total_comp'
}

gen t = elastic_salary_sum
replace t = . if t == 0
winsor t, gen(`elastic_salary_var') p(0.01)
drop t

gen t = elastic_weight_sum
replace t = . if t == 0
winsor t, gen(`elastic_weight_var') p(0.01)
drop t

gen t = elastic_total_comp_sum
replace t = . if t == 0
winsor t, gen(`elastic_total_comp_var') p(0.01)
drop t

gen t = elastic_salary_sum/total_wage_bill
replace t = . if t == 0
winsor t, gen(`elastic_per_salary_var') p(0.01)
drop t

gen t = elastic_weight_sum/total_employment
replace t = . if t == 0
winsor t, gen(`elastic_per_weight_var') p(0.01)
drop t

gen t = elastic_total_comp_sum/total_compensation_bill
replace t = . if t == 0
winsor t, gen(`elastic_per_total_comp_var') p(0.01)
drop t

gen emp_cstat2wfd_2 = emp/total_employment
drop if emp_cstat2wfd_2 <= 0.25
drop if emp_cstat2wfd_2 > 2
drop emp_cstat2wfd_2

sum total_employment if xsalesmktrawCIQ != ., detail
local perc = r(p25)

* 7-12
regress w_xsalesmktrawCIQ `elastic_salary_var', r
post `posth' (7) ("`sm_type'") (_b[`elastic_salary_var']) (e(r2))

regress w_xsalesmktrawCIQ `elastic_total_comp_var', r
post `posth' (8) ("`sm_type'") (_b[`elastic_total_comp_var']) (e(r2))

regress w_per_xsalesmktrawCIQ `elastic_per_salary_var', r
post `posth' (9) ("`sm_type'") (_b[`elastic_per_salary_var']) (e(r2))

regress w_per_xsalesmktrawCIQ `elastic_per_total_comp_var', r
post `posth' (10) ("`sm_type'") (_b[`elastic_per_total_comp_var']) (e(r2))

regress w_xsalesmktrawCIQ `elastic_weight_var', r
post `posth' (11) ("`sm_type'") (_b[`elastic_weight_var']) (e(r2))

regress w_per_xsalesmktrawCIQ `elastic_per_weight_var', r
post `posth' (12) ("`sm_type'") (_b[`elastic_per_weight_var']) (e(r2))

* 19-24
regress w_xsalesmktrawCIQ `elastic_salary_var' if total_employment < `perc', r
post `posth' (19) ("`sm_type'") (_b[`elastic_salary_var']) (e(r2))

regress w_xsalesmktrawCIQ `elastic_total_comp_var' if total_employment < `perc', r
post `posth' (20) ("`sm_type'") (_b[`elastic_total_comp_var']) (e(r2))

regress w_per_xsalesmktrawCIQ `elastic_per_salary_var' if total_employment < `perc', r
post `posth' (21) ("`sm_type'") (_b[`elastic_per_salary_var']) (e(r2))

regress w_per_xsalesmktrawCIQ `elastic_per_total_comp_var' if total_employment < `perc', r
post `posth' (22) ("`sm_type'") (_b[`elastic_per_total_comp_var']) (e(r2))

regress w_xsalesmktrawCIQ `elastic_weight_var' if total_employment < `perc', r
post `posth' (23) ("`sm_type'") (_b[`elastic_weight_var']) (e(r2))

regress w_per_xsalesmktrawCIQ `elastic_per_weight_var' if total_employment < `perc', r
post `posth' (24) ("`sm_type'") (_b[`elastic_per_weight_var']) (e(r2))

postclose `posth'

use `results_tmp', clear
sort regression_number
save "..\output\wfd_regressions_by_type_4_elastic_weight.dta", replace

di as text "Saved: ..\output\wfd_regressions_by_type_4_elastic_weight.dta"
