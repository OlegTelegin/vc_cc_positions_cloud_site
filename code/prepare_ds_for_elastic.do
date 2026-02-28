use "D:\cursor_projects\vc_cc_positions_cloud_site\data\temp_wfd_1702_6_w_certainty_quartiles.dta", replace

foreach var of varlist total_compensation_k1000_* {
	replace `var' = . if `var' == 0
}

foreach var of varlist total_compensation_k1000_* {
    quietly count if !missing(`var')
    if r(N) > 100 {
        winsor `var', gen(w_`var') p(0.01)
    }
    else {
        local vtype : type `var'
        gen `vtype' w_`var' = `var'
    }
}

foreach var of varlist total_compensation_k1000_* w_total_compensation_k1000_* {
	replace `var' = 0 if `var' == .
}

save "D:\cursor_projects\vc_cc_positions_cloud_site\data\temp_wfd_1702_6_w_certainty_quartiles_for_elastic.dta", replace


use "D:\cursor_projects\vc_cc_positions_cloud_site\data\temp_wfd_1702_6_w_certainty_quartiles.dta", replace

foreach var of varlist salary_k1000_* {
	replace `var' = . if `var' == 0
}

foreach var of varlist salary_k1000_* {
    quietly count if !missing(`var')
    if r(N) > 100 {
        winsor `var', gen(w_`var') p(0.01)
    }
    else {
        local vtype : type `var'
        gen `vtype' w_`var' = `var'
    }
}

foreach var of varlist salary_k1000_* w_salary_k1000_* {
	replace `var' = 0 if `var' == .
}

save "D:\cursor_projects\vc_cc_positions_cloud_site\data\temp_wfd_1702_6_w_certainty_quartiles_for_elastic_salary.dta", replace

