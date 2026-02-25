clear all
set more off

* Input / output
local input_data  "D:\cursor_projects\vc_cc_positions_cloud_site\data\full_data_to_build_buckets.dta"
local output_data "D:\cursor_projects\vc_cc_positions_cloud_site\data\certainty_quartile_by_naics.dta"

use "`input_data'", clear

* Safety checks
capture confirm variable abs_ln_emp_cstat2wfd_2
if _rc {
    di as error "Variable abs_ln_emp_cstat2wfd_2 not found."
    exit 111
}

capture confirm variable naics2
if _rc {
    di as error "Variable naics2 not found."
    exit 111
}

tempfile naics2_bucket_map

preserve
    keep naics2 abs_ln_emp_cstat2wfd_2
    drop if missing(naics2) | missing(abs_ln_emp_cstat2wfd_2)

    * Build one row per naics2 with its size and central tendency.
    * Sorting by mean(abs_ln_emp_cstat2wfd_2) gives an ordered scale;
    * cumulative counts then create near-equal observation buckets.
    collapse ///
        (count) naics2_obs = abs_ln_emp_cstat2wfd_2 ///
        (mean)  naics2_mean_abs_ln = abs_ln_emp_cstat2wfd_2, ///
        by(naics2)

    gsort +naics2_mean_abs_ln +naics2
    egen long total_obs = total(naics2_obs)
    gen long cum_obs = sum(naics2_obs)
    gen double cum_share = cum_obs / total_obs

    gen byte abs_ln_emp_cstat2wfd_2_naics2_q4 = 1
    replace abs_ln_emp_cstat2wfd_2_naics2_q4 = 2 if cum_share > 0.25
    replace abs_ln_emp_cstat2wfd_2_naics2_q4 = 3 if cum_share > 0.50
    replace abs_ln_emp_cstat2wfd_2_naics2_q4 = 4 if cum_share > 0.75

    label define abs_ln_q4_lbl 1 "Q1 (lowest)" 2 "Q2" 3 "Q3" 4 "Q4 (highest)", replace
    label values abs_ln_emp_cstat2wfd_2_naics2_q4 abs_ln_q4_lbl
    label var abs_ln_emp_cstat2wfd_2_naics2_q4 ///
        "Quartile of abs_ln_emp_cstat2wfd_2 by naics2 (naics2 kept intact)"

    keep naics2 abs_ln_emp_cstat2wfd_2_naics2_q4
    save "`naics2_bucket_map'", replace
restore

merge m:1 naics2 using "`naics2_bucket_map'", nogen keep(master match)

* Create variables with global quartile thresholds for total_employment
capture confirm variable total_employment
if _rc {
    di as error "Variable total_employment not found."
    exit 111
}

quietly _pctile total_employment if !missing(total_employment), p(25 50 75)
gen double total_employment_q25 = r(r1)
gen double total_employment_q50 = r(r2)
gen double total_employment_q75 = r(r3)

label var total_employment_q25 "25th percentile threshold of total_employment"
label var total_employment_q50 "50th percentile threshold of total_employment"
label var total_employment_q75 "75th percentile threshold of total_employment"

* Quick diagnostics in log
tab abs_ln_emp_cstat2wfd_2_naics2_q4, missing

preserve
    keep if !missing(abs_ln_emp_cstat2wfd_2_naics2_q4)
    contract abs_ln_emp_cstat2wfd_2_naics2_q4
    list, clean noobs
restore

keep naics2 abs_ln_emp_cstat2wfd_2_naics2_q4 total_employment_q25 total_employment_q50 total_employment_q75
collapse (mean) abs_ln_emp_cstat2wfd_2_naics2_q4 total_employment_q25 total_employment_q50 total_employment_q75, by(naics2)
rename abs_ln_emp_cstat2wfd_2_naics2_q4 certainty_quartile_by_naics

save "`output_data'", replace

di as result "Saved file with quartile bucket: `output_data'"
