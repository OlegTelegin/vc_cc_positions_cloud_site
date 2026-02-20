clear all
set more off

local sm_type "amir_all_roles"

tempfile results_tmp
tempname posth

* Load role mapping (all role numbers are used together)
import delimited "..\web\data\amir_category_role_numbers.csv", clear varnames(1)

capture confirm variable role_k1000_v3_num
if _rc {
    di as error "CSV is missing required column: role_k1000_v3_num"
    exit 111
}

drop if missing(role_k1000_v3_num)
quietly count
if r(N) == 0 {
    di as error "No category-role mapping rows found in ..\web\data\amir_category_role_numbers.csv"
    exit 2000
}

levelsof role_k1000_v3_num, local(all_ids)
if "`all_ids'" == "" {
    di as error "No valid role IDs found in ..\web\data\amir_category_role_numbers.csv"
    exit 2000
}

postfile `posth' ///
    int regression_number ///
    str40 sm_type ///
    double coefficient ///
    double r2 ///
    using `results_tmp', replace

* -------- First sample (full data) --------
use "..\data\temp_wfd_1702_5.dta", clear

gen double rhs_salary_sum = 0
gen double rhs_weight_sum = 0
gen double rhs_tcomp_sum = 0

foreach id of local all_ids {
    capture confirm variable salary_k1000_`id'
    if _rc {
        di as error "Variable salary_k1000_`id' not found in temp_wfd_1702_5.dta"
        exit 111
    }
    capture confirm variable weight_k1000_`id'
    if _rc {
        di as error "Variable weight_k1000_`id' not found in temp_wfd_1702_5.dta"
        exit 111
    }
    capture confirm variable total_compensation_k1000_`id'
    if _rc {
        di as error "Variable total_compensation_k1000_`id' not found in temp_wfd_1702_5.dta"
        exit 111
    }

    replace rhs_salary_sum = rhs_salary_sum + salary_k1000_`id'
    replace rhs_weight_sum = rhs_weight_sum + weight_k1000_`id'
    replace rhs_tcomp_sum = rhs_tcomp_sum + total_compensation_k1000_`id'
}

gen t = rhs_salary_sum
replace t = . if t == 0
winsor t, gen(rhs_salary) p(0.01)
drop t

gen t = rhs_weight_sum
replace t = . if t == 0
winsor t, gen(rhs_weight) p(0.01)
drop t

gen t = rhs_tcomp_sum
replace t = . if t == 0
winsor t, gen(rhs_tcomp) p(0.01)
drop t

gen t = rhs_salary_sum/total_wage_bill
replace t = . if t == 0
winsor t, gen(rhs_per_salary) p(0.01)
drop t

gen t = rhs_weight_sum/total_employment
replace t = . if t == 0
winsor t, gen(rhs_per_weight) p(0.01)
drop t

gen t = rhs_tcomp_sum/total_compensation_bill
replace t = . if t == 0
winsor t, gen(rhs_per_tcomp) p(0.01)
drop t

sum total_employment if xsalesmktrawCIQ != ., detail
local perc = r(p25)

* 1-6
regress w_xsalesmktrawCIQ rhs_salary, r
post `posth' (1) ("`sm_type'") (_b[rhs_salary]) (e(r2))

regress w_xsalesmktrawCIQ rhs_tcomp, r
post `posth' (2) ("`sm_type'") (_b[rhs_tcomp]) (e(r2))

regress w_per_xsalesmktrawCIQ rhs_per_salary, r
post `posth' (3) ("`sm_type'") (_b[rhs_per_salary]) (e(r2))

regress w_per_xsalesmktrawCIQ rhs_per_tcomp, r
post `posth' (4) ("`sm_type'") (_b[rhs_per_tcomp]) (e(r2))

regress w_xsalesmktrawCIQ rhs_weight, r
post `posth' (5) ("`sm_type'") (_b[rhs_weight]) (e(r2))

regress w_per_xsalesmktrawCIQ rhs_per_weight, r
post `posth' (6) ("`sm_type'") (_b[rhs_per_weight]) (e(r2))

* 13-18
regress w_xsalesmktrawCIQ rhs_salary if total_employment < `perc', r
post `posth' (13) ("`sm_type'") (_b[rhs_salary]) (e(r2))

regress w_xsalesmktrawCIQ rhs_tcomp if total_employment < `perc', r
post `posth' (14) ("`sm_type'") (_b[rhs_tcomp]) (e(r2))

regress w_per_xsalesmktrawCIQ rhs_per_salary if total_employment < `perc', r
post `posth' (15) ("`sm_type'") (_b[rhs_per_salary]) (e(r2))

regress w_per_xsalesmktrawCIQ rhs_per_tcomp if total_employment < `perc', r
post `posth' (16) ("`sm_type'") (_b[rhs_per_tcomp]) (e(r2))

regress w_xsalesmktrawCIQ rhs_weight if total_employment < `perc', r
post `posth' (17) ("`sm_type'") (_b[rhs_weight]) (e(r2))

regress w_per_xsalesmktrawCIQ rhs_per_weight if total_employment < `perc', r
post `posth' (18) ("`sm_type'") (_b[rhs_per_weight]) (e(r2))

* -------- Second sample (filtered on emp_cstat2wfd_2) --------
use "..\data\temp_wfd_1702_5.dta", clear

gen double rhs_salary_sum = 0
gen double rhs_weight_sum = 0
gen double rhs_tcomp_sum = 0

foreach id of local all_ids {
    capture confirm variable salary_k1000_`id'
    if _rc {
        di as error "Variable salary_k1000_`id' not found in temp_wfd_1702_5.dta"
        exit 111
    }
    capture confirm variable weight_k1000_`id'
    if _rc {
        di as error "Variable weight_k1000_`id' not found in temp_wfd_1702_5.dta"
        exit 111
    }
    capture confirm variable total_compensation_k1000_`id'
    if _rc {
        di as error "Variable total_compensation_k1000_`id' not found in temp_wfd_1702_5.dta"
        exit 111
    }

    replace rhs_salary_sum = rhs_salary_sum + salary_k1000_`id'
    replace rhs_weight_sum = rhs_weight_sum + weight_k1000_`id'
    replace rhs_tcomp_sum = rhs_tcomp_sum + total_compensation_k1000_`id'
}

gen t = rhs_salary_sum
replace t = . if t == 0
winsor t, gen(rhs_salary) p(0.01)
drop t

gen t = rhs_weight_sum
replace t = . if t == 0
winsor t, gen(rhs_weight) p(0.01)
drop t

gen t = rhs_tcomp_sum
replace t = . if t == 0
winsor t, gen(rhs_tcomp) p(0.01)
drop t

gen t = rhs_salary_sum/total_wage_bill
replace t = . if t == 0
winsor t, gen(rhs_per_salary) p(0.01)
drop t

gen t = rhs_weight_sum/total_employment
replace t = . if t == 0
winsor t, gen(rhs_per_weight) p(0.01)
drop t

gen t = rhs_tcomp_sum/total_compensation_bill
replace t = . if t == 0
winsor t, gen(rhs_per_tcomp) p(0.01)
drop t

gen emp_cstat2wfd_2 = emp/total_employment
drop if emp_cstat2wfd_2 <= 0.25
drop if emp_cstat2wfd_2 > 2
drop emp_cstat2wfd_2

sum total_employment if xsalesmktrawCIQ != ., detail
local perc = r(p25)

* 7-12
regress w_xsalesmktrawCIQ rhs_salary, r
post `posth' (7) ("`sm_type'") (_b[rhs_salary]) (e(r2))

regress w_xsalesmktrawCIQ rhs_tcomp, r
post `posth' (8) ("`sm_type'") (_b[rhs_tcomp]) (e(r2))

regress w_per_xsalesmktrawCIQ rhs_per_salary, r
post `posth' (9) ("`sm_type'") (_b[rhs_per_salary]) (e(r2))

regress w_per_xsalesmktrawCIQ rhs_per_tcomp, r
post `posth' (10) ("`sm_type'") (_b[rhs_per_tcomp]) (e(r2))

regress w_xsalesmktrawCIQ rhs_weight, r
post `posth' (11) ("`sm_type'") (_b[rhs_weight]) (e(r2))

regress w_per_xsalesmktrawCIQ rhs_per_weight, r
post `posth' (12) ("`sm_type'") (_b[rhs_per_weight]) (e(r2))

* 19-24
regress w_xsalesmktrawCIQ rhs_salary if total_employment < `perc', r
post `posth' (19) ("`sm_type'") (_b[rhs_salary]) (e(r2))

regress w_xsalesmktrawCIQ rhs_tcomp if total_employment < `perc', r
post `posth' (20) ("`sm_type'") (_b[rhs_tcomp]) (e(r2))

regress w_per_xsalesmktrawCIQ rhs_per_salary if total_employment < `perc', r
post `posth' (21) ("`sm_type'") (_b[rhs_per_salary]) (e(r2))

regress w_per_xsalesmktrawCIQ rhs_per_tcomp if total_employment < `perc', r
post `posth' (22) ("`sm_type'") (_b[rhs_per_tcomp]) (e(r2))

regress w_xsalesmktrawCIQ rhs_weight if total_employment < `perc', r
post `posth' (23) ("`sm_type'") (_b[rhs_weight]) (e(r2))

regress w_per_xsalesmktrawCIQ rhs_per_weight if total_employment < `perc', r
post `posth' (24) ("`sm_type'") (_b[rhs_per_weight]) (e(r2))

postclose `posth'

use `results_tmp', clear
sort sm_type regression_number
save "..\output\wfd_regressions_by_type_3_amir_category.dta", replace

di as text "Saved: ..\output\wfd_regressions_by_type_3_amir_category.dta"
