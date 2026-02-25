use "D:\revelio_data\wfd\get_needed_salaries_data\temp_wfd_1702_5.dta", replace

merge n:1 rcid using "F:\revelio_data\company_naics_code_parts_all.dta", keep(1 3) nogen
gen naics2 = substr(naics_code, 1, 2)

merge n:1 naics2 using "D:\cursor_projects\vc_cc_positions_cloud_site\data\certainty_quartile_by_naics.dta", keep(1 3) nogen

* 1) Create quartiles for total_employment from the threshold variables
gen byte total_employment_q4 = .

replace total_employment_q4 = 1 if total_employment <  total_employment_q25
replace total_employment_q4 = 2 if total_employment >= total_employment_q25 ///
    & total_employment <  total_employment_q50
replace total_employment_q4 = 3 if total_employment >= total_employment_q50 ///
    & total_employment <  total_employment_q75
replace total_employment_q4 = 4 if total_employment >= total_employment_q75 ///
    & !missing(total_employment)

label define q4lbl 1 "Q1" 2 "Q2" 3 "Q3" 4 "Q4"
label values total_employment_q4 q4lbl

* 2) Create 16 quartile-by-quartile buckets
gen byte bucket16 = 4*(certainty_quartile_by_naics - 1) + total_employment_q4 ///
    if !missing(certainty_quartile_by_naics, total_employment_q4)

label variable bucket16 "16 buckets: ceratinty by naics2 quartile x total_employment quartile"

* Check
tab total_employment_q4
tab certainty_quartile_by_naics total_employment_q4, missing
tab bucket16

save "D:\revelio_data\wfd\get_needed_salaries_data\temp_wfd_1702_5_w_certainty_quartiles.dta", replace










use "D:\revelio_data\wfd\get_needed_salaries_data\temp_wfd_1702_6.dta", replace

merge n:1 rcid using "F:\revelio_data\company_naics_code_parts_all.dta", keep(1 3) nogen
gen naics2 = substr(naics_code, 1, 2)

merge n:1 naics2 using "D:\cursor_projects\vc_cc_positions_cloud_site\data\certainty_quartile_by_naics.dta", keep(1 3) nogen

* 1) Create quartiles for total_employment from the threshold variables
gen byte total_employment_q4 = .

replace total_employment_q4 = 1 if total_employment <  total_employment_q25
replace total_employment_q4 = 2 if total_employment >= total_employment_q25 ///
    & total_employment <  total_employment_q50
replace total_employment_q4 = 3 if total_employment >= total_employment_q50 ///
    & total_employment <  total_employment_q75
replace total_employment_q4 = 4 if total_employment >= total_employment_q75 ///
    & !missing(total_employment)

label define q4lbl 1 "Q1" 2 "Q2" 3 "Q3" 4 "Q4"
label values total_employment_q4 q4lbl

* 2) Create 16 quartile-by-quartile buckets
gen byte bucket16 = 4*(certainty_quartile_by_naics - 1) + total_employment_q4 ///
    if !missing(certainty_quartile_by_naics, total_employment_q4)

label variable bucket16 "16 buckets: ceratinty by naics2 quartile x total_employment quartile"

* Check
tab total_employment_q4
tab certainty_quartile_by_naics total_employment_q4, missing
tab bucket16

save "D:\revelio_data\wfd\get_needed_salaries_data\temp_wfd_1702_6_w_certainty_quartiles.dta", replace













