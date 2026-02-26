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

* save estimation sample
gen byte samp1 = e(sample)

* 2) Get fitted value from model 1
predict double xb1 if samp1, xb

* 3) Keep only the bucket16 contribution (NOT the constant, NOT salary)
gen double bucket_fix = xb1 - _b[_cons] - _b[w_salary_k10_8]*w_salary_k10_8 if samp1

* 4) Second regression, with bucket contribution fixed at coefficient 1
constraint drop _all
constraint 1 bucket_fix = 1

* 2
cnsreg w_xsalesmktrawCIQ w_total_compensation_k10_8 bucket_fix if samp1, constraints(1) vce(robust)

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
post `posth' (2) ("`sm_type'") (_b[w_total_compensation_k10_8]) (R2_c)

* 5
cnsreg w_xsalesmktrawCIQ w_weight_k10_8 bucket_fix if samp1, constraints(1) vce(robust)

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
post `posth' (5) ("`sm_type'") (_b[w_weight_k10_8]) (R2_c)

* 13
cnsreg w_xsalesmktrawCIQ w_salary_k10_8 bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

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
post `posth' (13) ("`sm_type'") (_b[w_salary_k10_8]) (R2_c)

* 14
cnsreg w_xsalesmktrawCIQ w_total_compensation_k10_8 bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

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
post `posth' (14) ("`sm_type'") (_b[w_total_compensation_k10_8]) (R2_c)

* 17
cnsreg w_xsalesmktrawCIQ w_weight_k10_8 bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

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
post `posth' (17) ("`sm_type'") (_b[w_weight_k10_8]) (R2_c)

* 3
regress w_per_xsalesmktrawCIQ w_per_salary_k10_8 i.bucket16, r
post `posth' (3) ("`sm_type'") (_b[w_per_salary_k10_8]) (e(r2))

drop samp1 bucket_fix xb1
* save estimation sample
gen byte samp1 = e(sample)

* 2) Get fitted value from model 1
predict double xb1 if samp1, xb

* 3) Keep only the bucket16 contribution (NOT the constant, NOT salary)
gen double bucket_fix = xb1 - _b[_cons] - _b[w_per_salary_k10_8]*w_per_salary_k10_8 if samp1

* 4) Second regression, with bucket contribution fixed at coefficient 1
constraint drop _all
constraint 1 bucket_fix = 1

* 4
cnsreg w_per_xsalesmktrawCIQ w_per_total_compensation_k10_8 bucket_fix if samp1, constraints(1) vce(robust)

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
post `posth' (4) ("`sm_type'") (_b[w_per_total_compensation_k10_8]) (R2_c)

* 6
cnsreg w_per_xsalesmktrawCIQ w_per_weight_k10_8 bucket_fix if samp1, constraints(1) vce(robust)

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
post `posth' (6) ("`sm_type'") (_b[w_per_weight_k10_8]) (R2_c)

* 15
cnsreg w_per_xsalesmktrawCIQ w_per_salary_k10_8 bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

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
post `posth' (15) ("`sm_type'") (_b[w_per_salary_k10_8]) (R2_c)

* 16
cnsreg w_per_xsalesmktrawCIQ w_per_total_compensation_k10_8 bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

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
post `posth' (16) ("`sm_type'") (_b[w_per_total_compensation_k10_8]) (R2_c)

* 18
cnsreg w_per_xsalesmktrawCIQ w_per_weight_k10_8 bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

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
post `posth' (18) ("`sm_type'") (_b[w_per_weight_k10_8]) (R2_c)

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

* save estimation sample
gen byte samp1 = e(sample)

* 2) Get fitted value from model 1
predict double xb1 if samp1, xb

* 3) Keep only the bucket16 contribution (NOT the constant, NOT salary)
gen double bucket_fix = xb1 - _b[_cons] - _b[w_salary_k10_8]*w_salary_k10_8 if samp1

* 4) Second regression, with bucket contribution fixed at coefficient 1
constraint drop _all
constraint 1 bucket_fix = 1

* 8
cnsreg w_xsalesmktrawCIQ w_total_compensation_k10_8 bucket_fix if samp1, constraints(1) vce(robust)

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
post `posth' (8) ("`sm_type'") (_b[w_total_compensation_k10_8]) (R2_c)

* 11
cnsreg w_xsalesmktrawCIQ w_weight_k10_8 bucket_fix if samp1, constraints(1) vce(robust)

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
post `posth' (11) ("`sm_type'") (_b[w_weight_k10_8]) (R2_c)

* 19
cnsreg w_xsalesmktrawCIQ w_salary_k10_8 bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

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
post `posth' (19) ("`sm_type'") (_b[w_salary_k10_8]) (R2_c)

* 20
cnsreg w_xsalesmktrawCIQ w_total_compensation_k10_8 bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

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
post `posth' (20) ("`sm_type'") (_b[w_total_compensation_k10_8]) (R2_c)

* 23
cnsreg w_xsalesmktrawCIQ w_weight_k10_8 bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

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
post `posth' (23) ("`sm_type'") (_b[w_weight_k10_8]) (R2_c)

* 9
regress w_per_xsalesmktrawCIQ w_per_salary_k10_8 i.bucket16, r
post `posth' (9) ("`sm_type'") (_b[w_per_salary_k10_8]) (e(r2))

drop samp1 bucket_fix xb1
* save estimation sample
gen byte samp1 = e(sample)

* 2) Get fitted value from model 1
predict double xb1 if samp1, xb

* 3) Keep only the bucket16 contribution (NOT the constant, NOT salary)
gen double bucket_fix = xb1 - _b[_cons] - _b[w_per_salary_k10_8]*w_per_salary_k10_8 if samp1

* 4) Second regression, with bucket contribution fixed at coefficient 1
constraint drop _all
constraint 1 bucket_fix = 1

* 10
cnsreg w_per_xsalesmktrawCIQ w_per_total_compensation_k10_8 bucket_fix if samp1, constraints(1) vce(robust)

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
post `posth' (10) ("`sm_type'") (_b[w_per_total_compensation_k10_8]) (R2_c)

* 12
cnsreg w_per_xsalesmktrawCIQ w_per_weight_k10_8 bucket_fix if samp1, constraints(1) vce(robust)

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
post `posth' (12) ("`sm_type'") (_b[w_per_weight_k10_8]) (R2_c)

* 21
cnsreg w_per_xsalesmktrawCIQ w_per_salary_k10_8 bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

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
post `posth' (21) ("`sm_type'") (_b[w_per_salary_k10_8]) (R2_c)

* 22
cnsreg w_per_xsalesmktrawCIQ w_per_total_compensation_k10_8 bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

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
post `posth' (22) ("`sm_type'") (_b[w_per_total_compensation_k10_8]) (R2_c)

* 24
cnsreg w_per_xsalesmktrawCIQ w_per_weight_k10_8 bucket_fix if samp1 & total_employment < `perc', constraints(1) vce(robust)

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
post `posth' (24) ("`sm_type'") (_b[w_per_weight_k10_8]) (R2_c)

postclose `posth'

use `results_tmp', clear
sort regression_number
save "..\output\wfd_regressions_by_type_17_fe.dta", replace
