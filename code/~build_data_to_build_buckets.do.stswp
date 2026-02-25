use "D:\revelio_data\wfd\get_needed_salaries_data\temp_wfd_1702_5.dta", replace

merge n:1 rcid using "F:\revelio_data\company_naics_code_parts_all.dta", keep(1 3) nogen
keep rcid year emp naicsh_3 total_employment naics_code
drop naicsh_3

gen emp_cstat2wfd_2=emp/total_employment
gen ln_emp_cstat2wfd_2 = ln(emp_cstat2wfd_2)

scatter ln_emp_cstat2wfd_2 total_employment
scatter ln_emp_cstat2wfd_2 total_employment if total_employment < 2

* gen naics3 = substr(naics_code, 1, 3)
* tab naics3

gen naics2 = substr(naics_code, 1, 2)
tab naics2

gen abs_ln_emp_cstat2wfd_2 = abs(ln_emp_cstat2wfd_2)

save "D:\cursor_projects\vc_cc_positions_cloud_site\data\full_data_to_build_buckets.dta", replace
use "D:\cursor_projects\vc_cc_positions_cloud_site\data\full_data_to_build_buckets.dta", replace





























