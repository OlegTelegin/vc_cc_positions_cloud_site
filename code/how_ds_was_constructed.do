

use "D:\revelio_data\wfd\get_needed_salaries_data\wide_file_1.dta", replace
append using "D:\revelio_data\wfd\get_needed_salaries_data\wide_file_2.dta"

merge 1:1 rcid year using "F:\revelio_data\try_to_match_hms\FYrevelioWfD_HMS_together.dta"
keep if _merge == 3
drop _merge

merge 1:1 rcid year using "D:\revelio_data\wfd\get_needed_salaries_data\total_employment_denominator_together.dta", keep(1 3) nogen

merge 1:1 rcid year using "D:\revelio_data\wfd\get_needed_salaries_data\total_wage_bill_denominator_together.dta", keep(1 3) nogen

merge 1:1 rcid year using "D:\revelio_data\wfd\get_needed_salaries_data\total_compensation_bill_denominator_together.dta", keep(1 3) nogen

rename (total_employment total_wage_bill total_compensation_bill) (total_employment_s total_wage_bill_s total_compensation_bill_s)
merge 1:1 rcid year using "D:\revelio_data\wfd\get_needed_salaries_data\denominators_not_needed.dta", keep(1 3) nogen
replace total_employment = total_employment_s if total_employment_s !=.
replace total_wage_bill = total_wage_bill_s if total_wage_bill_s !=.
replace total_compensation_bill = total_compensation_bill_s if total_compensation_bill_s !=.
drop total_employment_s total_wage_bill_s total_compensation_bill_s

keep if xsalesmkt~=.
* keep if emp~=.
* keep if C_tot~=.

* Move counts to thousands and money to thousands dollars
foreach var of varlist weight_k10_* salary_k10_* total_compensation_k10_* weight_k50_* salary_k50_* total_compensation_k50_* {
	replace `var' = `var' / 1000
}
foreach var of varlist weight_k1000_* salary_k1000_* total_compensation_k1000_* {
	replace `var' = `var' / 1000
}

replace total_employment = total_employment / 1000
replace total_wage_bill = total_wage_bill / 1000
replace total_compensation_bill = total_compensation_bill / 1000

* Move Cap IQ money to thousands dollars 
replace xsalesmktrawCIQ = xsalesmktrawCIQ * 1000
replace xsalesmktCIQ = xsalesmktCIQ * 1000
replace xsalesmkt = xsalesmkt * 1000

* ???????????????????????????????????????
drop if total_employment == .

foreach var of varlist weight_k10_* weight_k50_* {
	gen t=`var'/total_employment
	replace t = . if t == 0
	winsor t, gen(w_per_`var') p(0.01)
	drop t
}

foreach var of varlist salary_k10_* salary_k50_* {
	gen t=`var'/total_wage_bill
	replace t = . if t == 0
	winsor t, gen(w_per_`var') p(0.01)
	drop t
}

foreach var of varlist total_compensation_k10_* total_compensation_k50_* {
	gen t=`var'/total_compensation_bill
	replace t = . if t == 0
	winsor t, gen(w_per_`var') p(0.01)
	drop t
}

foreach var of varlist weight_k10_* weight_k50_* salary_k10_* salary_k50_* total_compensation_k10_* total_compensation_k50_* {
	gen t=`var'
	replace t = . if t == 0
	winsor t, gen(w_`var') p(0.01)
	drop t
}

/*
foreach var of varlist weight_k1000_* salary_k1000_* total_compensation_k1000_* {
    quietly count if `var' < .
    if (r(N) > 0) {
        gen t = `var' / total_employment

        if (r(N) < 100) {
            gen cnt_`var' = t
        }
        else {
            winsor t, gen(cnt_`var') p(0.01)
        }

        drop t
    }
}
*/

gen t=xsalesmktrawCIQ/total_employment
winsor t, gen(w_per_xsalesmktrawCIQ) p(0.01)
drop t

winsor xsalesmktrawCIQ, gen(w_xsalesmktrawCIQ) p(0.01)
gen dop_denom = 1

save "D:\revelio_data\wfd\get_needed_salaries_data\temp_wfd_1702_5.dta", replace

use "D:\revelio_data\wfd\get_needed_salaries_data\temp_wfd_1702_5.dta", replace

gen emp_cstat2wfd_2=emp/total_employment
drop if emp_cstat2wfd_2<=0.25
drop if emp_cstat2wfd_2>2
drop emp_cstat2wfd_2
save "D:\revelio_data\wfd\get_needed_salaries_data\temp_wfd_1702_6.dta", replace

