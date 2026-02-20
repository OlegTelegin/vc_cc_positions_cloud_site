use "..\data\temp_wfd_1702_5.dta", replace

sum total_employment if xsalesmktrawCIQ != ., detail
local perc = r(p25)

* 1
regress w_xsalesmktrawCIQ w_salary_k10_8, r
* 2
regress w_xsalesmktrawCIQ w_total_compensation_k10_8, r
* 3
regress w_per_xsalesmktrawCIQ w_per_salary_k10_8, r
* 4
regress w_per_xsalesmktrawCIQ w_per_total_compensation_k10_8, r
* 5
regress w_xsalesmktrawCIQ w_weight_k10_8, r
* 6
regress w_per_xsalesmktrawCIQ w_per_weight_k10_8, r



* 13
regress w_xsalesmktrawCIQ w_salary_k10_8 if total_employment < `perc', r
* 14
regress w_xsalesmktrawCIQ w_total_compensation_k10_8 if total_employment < `perc', r
* 15
regress w_per_xsalesmktrawCIQ w_per_salary_k10_8 if total_employment < `perc', r
* 16
regress w_per_xsalesmktrawCIQ w_per_total_compensation_k10_8 if total_employment < `perc', r
* 17
regress w_xsalesmktrawCIQ w_weight_k10_8 if total_employment < `perc', r
* 18
regress w_per_xsalesmktrawCIQ w_per_weight_k10_8 if total_employment < `perc', r






use "..\data\temp_wfd_1702_5.dta", replace

gen emp_cstat2wfd_2=emp/total_employment
drop if emp_cstat2wfd_2<=0.25
drop if emp_cstat2wfd_2>2
drop emp_cstat2wfd_2

sum total_employment if xsalesmktrawCIQ != ., detail
local perc = r(p25)

* 7
regress w_xsalesmktrawCIQ w_salary_k10_8, r
* 8
regress w_xsalesmktrawCIQ w_total_compensation_k10_8, r
* 9
regress w_per_xsalesmktrawCIQ w_per_salary_k10_8, r
* 10
regress w_per_xsalesmktrawCIQ w_per_total_compensation_k10_8, r
* 11
regress w_xsalesmktrawCIQ w_weight_k10_8, r
* 12
regress w_per_xsalesmktrawCIQ w_per_weight_k10_8, r



* 19
regress w_xsalesmktrawCIQ w_salary_k10_8 if total_employment < `perc', r
* 20
regress w_xsalesmktrawCIQ w_total_compensation_k10_8 if total_employment < `perc', r
* 21
regress w_per_xsalesmktrawCIQ w_per_salary_k10_8 if total_employment < `perc', r
* 22
regress w_per_xsalesmktrawCIQ w_per_total_compensation_k10_8 if total_employment < `perc', r
* 23
regress w_xsalesmktrawCIQ w_weight_k10_8 if total_employment < `perc', r
* 24
regress w_per_xsalesmktrawCIQ w_per_weight_k10_8 if total_employment < `perc', r






















