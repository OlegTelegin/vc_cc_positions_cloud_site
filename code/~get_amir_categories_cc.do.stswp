use "D:\cursor_projects\vc_cc_positions_cloud_site\data\1000_positions_num.dta", clear
replace role_k1000_v3=lower(role_k1000_v3)
***a. sales or marketing, less retail sales and retail sales associates
preserve
keep if strpos(role_k1000_v3, "sales") | strpos(role_k1000_v3, "mark")
drop if role_k1000_v3=="retail sales" /*658*/
drop if role_k1000_v3=="retail sales associate" /*816*/
levelsof role_k1000_v3_num, local(slsmkt)
restore
***b. retail sales and retail sales associates
preserve
keep if strpos(role_k1000_v3, "sales") | strpos(role_k1000_v3, "mark")
keep if role_k1000_v3=="retail sales" | role_k1000_v3=="retail sales associate" 
/*658*/
/*816*/
levelsof role_k1000_v3_num, local(retail)
restore
***c. customer service, not already in a
preserve
keep if strpos(role_k1000_v3, "customer")
foreach num of numlist `slsmkt'{
	quietly drop if role_k1000_v3_num==`num'
}
levelsof role_k1000_v3_num, local(customer)
restore

*** Export flagged role numbers for web app usage
tempfile amir_categories_tmp
postfile amirpost str64 category int role_k1000_v3_num using `amir_categories_tmp', replace

foreach num of local slsmkt {
	post amirpost ("sales_or_marketing_non_retail") (`num')
}

foreach num of local retail {
	post amirpost ("retail_sales") (`num')
}

foreach num of local customer {
	post amirpost ("customer_service_non_sales_marketing") (`num')
}

postclose amirpost

use `amir_categories_tmp', clear
sort category role_k1000_v3_num
export delimited using "D:\cursor_projects\vc_cc_positions_cloud_site\web\data\amir_category_role_numbers.csv", replace
