clear all
set more off

local sm_type "8th_out_of_10_classes"

tempfile results_tmp
tempname posth

postfile `posth' ///
    int regression_number ///
    str40 sm_type ///
    double coefficient ///
    double r2 ///
    using `results_tmp', replace

* -------- First sample (full data) --------
use "..\data\temp_wfd_1702_5_w_certainty_quartiles.dta", clear

sum total_employment if xsalesmktrawCIQ != ., detail
local perc = r(p25)

* 1
regress w_xsalesmktrawCIQ w_salary_k10_8 i.bucket16, r
post `posth' (1) ("`sm_type'") (_b[w_salary_k10_8]) (e(r2))

* 2
regress w_xsalesmktrawCIQ w_total_compensation_k10_8 i.bucket16, r
post `posth' (2) ("`sm_type'") (_b[w_total_compensation_k10_8]) (e(r2))

* 3
regress w_per_xsalesmktrawCIQ w_per_salary_k10_8 i.bucket16, r
post `posth' (3) ("`sm_type'") (_b[w_per_salary_k10_8]) (e(r2))

* 4
regress w_per_xsalesmktrawCIQ w_per_total_compensation_k10_8 i.bucket16, r
post `posth' (4) ("`sm_type'") (_b[w_per_total_compensation_k10_8]) (e(r2))

* 5
regress w_xsalesmktrawCIQ w_weight_k10_8 i.bucket16, r
post `posth' (5) ("`sm_type'") (_b[w_weight_k10_8]) (e(r2))

* 6
regress w_per_xsalesmktrawCIQ w_per_weight_k10_8 i.bucket16, r
post `posth' (6) ("`sm_type'") (_b[w_per_weight_k10_8]) (e(r2))

* 13
regress w_xsalesmktrawCIQ w_salary_k10_8 i.bucket16 if total_employment < `perc', r
post `posth' (13) ("`sm_type'") (_b[w_salary_k10_8]) (e(r2))

* 14
regress w_xsalesmktrawCIQ w_total_compensation_k10_8 i.bucket16 if total_employment < `perc', r
post `posth' (14) ("`sm_type'") (_b[w_total_compensation_k10_8]) (e(r2))

* 15
regress w_per_xsalesmktrawCIQ w_per_salary_k10_8 i.bucket16 if total_employment < `perc', r
post `posth' (15) ("`sm_type'") (_b[w_per_salary_k10_8]) (e(r2))

* 16
regress w_per_xsalesmktrawCIQ w_per_total_compensation_k10_8 i.bucket16 if total_employment < `perc', r
post `posth' (16) ("`sm_type'") (_b[w_per_total_compensation_k10_8]) (e(r2))

* 17
regress w_xsalesmktrawCIQ w_weight_k10_8 i.bucket16 if total_employment < `perc', r
post `posth' (17) ("`sm_type'") (_b[w_weight_k10_8]) (e(r2))

* 18
regress w_per_xsalesmktrawCIQ w_per_weight_k10_8 i.bucket16 if total_employment < `perc', r
post `posth' (18) ("`sm_type'") (_b[w_per_weight_k10_8]) (e(r2))

* -------- Second sample (filtered on emp_cstat2wfd_2) --------
use "..\data\temp_wfd_1702_5_w_certainty_quartiles.dta", clear

gen emp_cstat2wfd_2 = emp/total_employment
drop if emp_cstat2wfd_2 <= 0.25
drop if emp_cstat2wfd_2 > 2
drop emp_cstat2wfd_2

sum total_employment if xsalesmktrawCIQ != ., detail
local perc = r(p25)

* 7
regress w_xsalesmktrawCIQ w_salary_k10_8 i.bucket16, r
post `posth' (7) ("`sm_type'") (_b[w_salary_k10_8]) (e(r2))

* 8
regress w_xsalesmktrawCIQ w_total_compensation_k10_8 i.bucket16, r
post `posth' (8) ("`sm_type'") (_b[w_total_compensation_k10_8]) (e(r2))

* 9
regress w_per_xsalesmktrawCIQ w_per_salary_k10_8 i.bucket16, r
post `posth' (9) ("`sm_type'") (_b[w_per_salary_k10_8]) (e(r2))

* 10
regress w_per_xsalesmktrawCIQ w_per_total_compensation_k10_8 i.bucket16, r
post `posth' (10) ("`sm_type'") (_b[w_per_total_compensation_k10_8]) (e(r2))

* 11
regress w_xsalesmktrawCIQ w_weight_k10_8 i.bucket16, r
post `posth' (11) ("`sm_type'") (_b[w_weight_k10_8]) (e(r2))

* 12
regress w_per_xsalesmktrawCIQ w_per_weight_k10_8 i.bucket16, r
post `posth' (12) ("`sm_type'") (_b[w_per_weight_k10_8]) (e(r2))

* 19
regress w_xsalesmktrawCIQ w_salary_k10_8 i.bucket16 if total_employment < `perc', r
post `posth' (19) ("`sm_type'") (_b[w_salary_k10_8]) (e(r2))

* 20
regress w_xsalesmktrawCIQ w_total_compensation_k10_8 i.bucket16 if total_employment < `perc', r
post `posth' (20) ("`sm_type'") (_b[w_total_compensation_k10_8]) (e(r2))

* 21
regress w_per_xsalesmktrawCIQ w_per_salary_k10_8 i.bucket16 if total_employment < `perc', r
post `posth' (21) ("`sm_type'") (_b[w_per_salary_k10_8]) (e(r2))

* 22
regress w_per_xsalesmktrawCIQ w_per_total_compensation_k10_8 i.bucket16 if total_employment < `perc', r
post `posth' (22) ("`sm_type'") (_b[w_per_total_compensation_k10_8]) (e(r2))

* 23
regress w_xsalesmktrawCIQ w_weight_k10_8 i.bucket16 if total_employment < `perc', r
post `posth' (23) ("`sm_type'") (_b[w_weight_k10_8]) (e(r2))

* 24
regress w_per_xsalesmktrawCIQ w_per_weight_k10_8 i.bucket16 if total_employment < `perc', r
post `posth' (24) ("`sm_type'") (_b[w_per_weight_k10_8]) (e(r2))

postclose `posth'

use `results_tmp', clear
sort regression_number
save "..\output\wfd_regressions_by_type_17_fe.dta", replace
